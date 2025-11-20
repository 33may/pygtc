import uuid
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import time

import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from loguru import logger

from research.util.data import load_data, load_data_offline
from research.util.strategy_base import BaseStrategy, BaseStrategyManager
from research.util.Interfaces import StockOperation, MarketPointData
from research.util.ploting import plot_spread
from research.util.managers import TradeManager

# rolling stats for fixed-size window
class RollingWindowStats:
    """fixed-size rolling mean/std with O(1) update using sum and sumsq."""

    def __init__(self, win: int):
        assert win > 1, "win must be > 1"
        self.win = win
        self.buf = deque()
        self.sum = 0.0
        self.sumsq = 0.0

    def _pop_oldest(self):
        """remove oldest value when buffer is full."""
        old = self.buf.popleft()
        self.sum -= old
        self.sumsq -= old * old

    def push(self, x: float) -> Tuple[Optional[float], Optional[float], bool]:
        """
        add new value. returns (mean, std, filled)
        - mean/std are for current window if filled, else None
        - filled == True when len(buf) == win
        """
        # append
        self.buf.append(x)
        self.sum += x
        self.sumsq += x * x

        # trim
        if len(self.buf) > self.win:
            self._pop_oldest()

        filled = len(self.buf) == self.win
        if not filled:
            return None, None, False

        n = float(self.win)
        mean = self.sum / n
        # sample std (ddof=1), avoid negative due to fp errors
        var = max((self.sumsq - self.sum * self.sum / n) / (n - 1.0), 0.0)
        std = np.sqrt(var)
        return mean, std, True

    def warmup(self, values: pd.Series | List[float]):
        """fill buffer from iterable up to win (last win elements are used)."""
        self.buf.clear()
        self.sum = 0.0
        self.sumsq = 0.0
        if isinstance(values, pd.Series):
            values = values.dropna().to_list()
        # take only the tail window
        tail = values[-self.win:]
        for v in tail:
            self.buf.append(float(v))
            self.sum += float(v)
            self.sumsq += float(v) * float(v)

    def ready(self) -> bool:
        return len(self.buf) == self.win

    def current(self) -> Tuple[Optional[float], Optional[float]]:
        """return (mean, std) for current window if ready, else (None, None)."""
        if not self.ready():
            return None, None
        n = float(self.win)
        mean = self.sum / n
        var = max((self.sumsq - self.sum * self.sum / n) / (n - 1.0), 0.0)
        std = np.sqrt(var)
        return mean, std

@dataclass
class RollingState:
    rolling_mean: float
    rolling_std: float


@dataclass
class Target:
    ts: pd.Timestamp
    exposure: int # -1 (short spread), 0, +1 (long spread)
    z: float
    alpha: float
    beta: float


def regression(s1_data, s2_data):
    X = s1_data.values
    y = s2_data.values

    X_with_const = sm.add_constant(X)

    model = sm.OLS(y, X_with_const).fit()

    alpha, beta = model.params
    residuals = y - (alpha + beta * X)

    if np.isnan(residuals).any():
        return np.nan, np.nan, float(alpha), float(beta)

    adf_res = adfuller(residuals)

    return adf_res[0], adf_res[1], float(alpha), float(beta)


def compute_spread(s1, s2, alpha, beta):
    return s1 - (alpha + beta * s2)


class CPMRStrategy(BaseStrategy):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.z_entry = config["z_entry"]
        self.z_exit = config["z_exit"]
        self.max_hold = config["maxhold"]
        self.win_len = config["win_len"]

        self.state = RollingState(
            rolling_mean=0, rolling_std=0
        )

        self._roll = RollingWindowStats(self.win_len)

        self.alpha = None
        self.beta = None

        self.active_position = None

    def process(self, market_data_s1: MarketPointData, market_data_s2: MarketPointData):
        """
        process market data point

        args:
            market_data_s1: MarketPointData with ts and prices array
            market_data_s2: MarketPointData with ts and prices array

        returns:
            Target signal or None
        """
        # extract prices from internal data structure
        data_s1 = float(market_data_s1.price)
        data_s2 = float(market_data_s2.price)

        spread = compute_spread(data_s1, data_s2, self.alpha, self.beta)

        mean, std, filled = self._roll.push(spread)

        if not filled or mean is None or std == 0:
            return None

        z_score = (spread - mean) / std
        ts = market_data_s1.ts

        if self.active_position:
            if (
                    (self.active_position == "long" and z_score > 0)
                    or
                    (self.active_position == "short" and z_score < 0)
            ):
                self.active_position = None
                return Target(ts=ts, exposure=0, z=z_score, alpha=self.alpha, beta=self.beta)
            return None
        else:
            if z_score < self.z_entry:
                self.active_position = "long"
                return Target(ts=ts, exposure=+1, z=z_score, alpha=self.alpha, beta=self.beta)
            elif z_score > self.z_exit:
                self.active_position = "short"
                return Target(ts=ts, exposure=-1, z=z_score, alpha=self.alpha, beta=self.beta)
            else:
                return None

    def _check_status(self, data):
        # data is DataFrame with columns [ticker1, ticker2]
        s1_data = data.iloc[:, 0]
        s2_data = data.iloc[:, 1]

        adf_stat, p_val, alpha, beta = regression(s1_data, s2_data)

        return p_val, alpha, beta
    

    def _load_history(self, data) -> None:
        """
        warm up rolling window using last win_len spreads from provided data.
        - data: DataFrame with columns [ticker1, ticker2] (aligned by index).
        """
        # get data by column position
        s1 = data.iloc[:, 0].astype(float)
        s2 = data.iloc[:, 1].astype(float)
        s1, s2 = s1.align(s2, join="inner")

        # build full spread series and warm up last win points
        spread_series = compute_spread(s1, s2, self.alpha, self.beta)
        self._roll.warmup(spread_series)

        # update state
        mean, std = self._roll.current()
        self.state.rolling_mean = 0.0 if mean is None else float(mean)
        self.state.rolling_std = 0.0 if std is None else float(std)

    
    def initialize(self, data):
        p_val, self.alpha, self.beta = self._check_status(data)

        "TODO think if we want single init that return all stats and manager handle them as single init script, or just keep"



class CPMRManager(BaseStrategyManager):
    """
    manager for cointegrated pairs mean reversion strategy
    handles: data loading, initialization, execution, logging, plotting
    """

    def __init__(self, pair: List[str], strategy_config: Dict, base_units: float = 1.0):
        """
        initialize manager with pair and strategy configuration

        args:
            pair: list of two tickers
            strategy_config: dict with z_entry, z_exit, win_len, etc
            base_units: base position size
        """
        self.pair = pair
        self.strategy_config = strategy_config
        self.base_units = base_units

        # create strategy instance
        self.strat = CPMRStrategy(name=f"CPMR_{pair[0]}_{pair[1]}", config=strategy_config)

        # logging
        self.operations = []
        self.signals = []

        # trade management
        self.trade_manager = TradeManager()

        # data tracking for plotting
        self.price_history = []  # [(ts, s1_price, s2_price, spread, z_score)]

    def initialize(self, fit_start: str, fit_bar: str = "1d"):
        """
        load historical data and initialize strategy
        - load data for cointegration test
        - check cointegration, get alpha/beta
        - warm up rolling windows

        args:
            fit_start: start date for fitting data
            fit_bar: bar interval (1d, 1h, etc)

        returns:
            p_val, alpha, beta from cointegration test
        """
        logger.info(f"loading fit data from {fit_start}...")
        fit_data = load_data(tickers=self.pair, start=fit_start, bar=fit_bar)["Close"]

        # fit_data = load_data_offline(tickers=self.pair, start=fit_start, bar=fit_bar)

        logger.info(f"checking cointegration...")
        p_val, alpha, beta = self.strat._check_status(fit_data)

        # save alpha/beta to strategy
        self.strat.alpha = alpha
        self.strat.beta = beta

        # display cointegration test results
        if p_val < 0.05:
            logger.success(f"✓ cointegration test PASSED (p-value: {p_val:.4f} < 0.05)")
        else:
            logger.warning(f"✗ cointegration test FAILED (p-value: {p_val:.4f} >= 0.05)")

        logger.info(f"regression coefficients: alpha={alpha:.4f}, beta={beta:.4f}")

        logger.info(f"warming up rolling window...")
        self.strat._load_history(fit_data)
        logger.success(f"initialization complete!\n")

        return p_val, alpha, beta

    def check_health(self, fit_start: str, fit_bar: str = "1d", show_plot: bool = True):
        """
        check cointegration health and visualize spread

        args:
            fit_start: start date for fit data
            fit_bar: bar interval
            show_plot: whether to display plot

        returns:
            dict with health metrics
        """
        logger.info(f"performing health check...")

        # load data
        fit_data = load_data(tickers=self.pair, start=fit_start, bar=fit_bar)["Close"]

        # fit_data = load_data_offline(tickers=self.pair, start=fit_start, bar=fit_bar)

        # check cointegration
        p_val, alpha, beta = self.strat._check_status(fit_data)

        # prepare data for plotting
        s1_data = fit_data.iloc[:, 0]
        s2_data = fit_data.iloc[:, 1]

        # display results
        logger.info(f"\n{'='*50}")
        logger.info(f"  COINTEGRATION HEALTH CHECK")
        logger.info(f"{'='*50}")
        logger.info(f"  Pair: {self.pair[0]} / {self.pair[1]}")
        # logger.info(f"  Period: {fit_start} to {s1_data.index[-1].date()}")
        logger.info(f"  Data points: {len(s1_data)}")
        logger.info(f"{'='*50}")

        if p_val < 0.05:
            logger.success(f"  ✓ PASSED: p-value = {p_val:.4f} < 0.05")
            status = "PASSED"
        else:
            logger.warning(f"  ✗ FAILED: p-value = {p_val:.4f} >= 0.05")
            status = "FAILED"

        logger.info(f"  Alpha: {alpha:.4f}")
        logger.info(f"  Beta: {beta:.4f}")
        logger.info(f"{'='*50}\n")

        # plot spread
        if show_plot:
            logger.info("displaying spread plot...")
            plot_spread(
                stock1=s1_data,
                stock2=s2_data,
                alpha=alpha,
                beta=beta,
                stock1_name=self.pair[0],
                stock2_name=self.pair[1]
            )

        return {
            "status": status,
            "p_value": p_val,
            "alpha": alpha,
            "beta": beta,
            "pair": self.pair,
            "data_points": len(s1_data)
        }

    def _get_current_position(self, ticker: str) -> float:
        """get current position for ticker from TradeManager."""
        position = 0.0
        for trade in self.trade_manager.current_trades:
            if trade.ticker == ticker:
                position += trade.exposure
        return position

    def process_point(self, market_data):
        """
        process single market data point through strategy
        this is called by external loop (backtest, live trading, etc)

        args:
            market_data: tuple of (timestamp, series) from load_data iterator

        returns:
            list of StockOperation to execute, or None
        """
        # convert raw data to internal MarketPointData
        ts, point_series = market_data

        point_close = point_series
        # point_close = point_series

        s1, s2 = MarketPointData(ts=ts, price=point_close.iloc[0]), MarketPointData(ts=ts, price=point_close.iloc[1])

        # Calculate spread and z-score for tracking
        s1_price = float(point_close.iloc[0])
        s2_price = float(point_close.iloc[1])
        spread = compute_spread(s1_price, s2_price, self.strat.alpha, self.strat.beta)

        # Calculate z-score if rolling stats are ready
        mean, std = self.strat._roll.current()
        z_score = (spread - mean) / std if mean is not None and std is not None and std > 0 else None

        # for final plot
        self.price_history.append((ts, s1_price, s2_price, spread, z_score))

        # get signal from strategy
        target = self.strat.process(market_data_s1=s1, market_data_s2=s2)

        if not target:
            return None

        
        self.signals.append(target)

        # convert target to stock operations
        ts = target.ts
        s1, s2 = self.pair[0], self.pair[1]

        # calculate desired positions based on exposure
        if target.exposure > 0:
            # long spread: buy s1, sell s2
            desired = {s1: self.base_units, s2: -target.beta * self.base_units}
        elif target.exposure < 0:
            # short spread: sell s1, buy s2
            desired = {s1: -self.base_units, s2: target.beta * self.base_units}
        else:
            # flatten
            desired = {s1: 0.0, s2: 0.0}

        # generate operations based on position delta and track with TradeManager
        ops = []
        ticker_to_price = {self.pair[0]: point_close.iloc[0], self.pair[1]: point_close.iloc[1]}

        for ticker in [s1, s2]:
            current = self._get_current_position(ticker)
            target_pos = desired[ticker]
            delta = target_pos - current

            if abs(delta) < 1e-9:
                continue

            side = +1 if delta > 0 else -1
            amount = abs(delta)
            current_price = ticker_to_price[ticker]

            # create stock operation
            op = StockOperation(
                id=uuid.uuid4(),
                ts=ts,
                ticker=ticker,
                exposure=side,
                amount=amount
            )
            ops.append(op)

            action = "BUY" if op.exposure == 1 else "SELL"
            logger.info(f"  {action} {op.amount:.2f} {op.ticker}")

            # track with TradeManager
            if target_pos == 0:
                # closing position - find and close the trade
                for trade in self.trade_manager.current_trades:
                    if trade.ticker == ticker:
                        market_point = MarketPointData(ts=ts, price=current_price)
                        closed_trade = self.trade_manager.close_trade(trade.uuid, market_point)
                        logger.info(f"  → Closed {ticker} trade | PnL: ${closed_trade.pnl:.2f}")
                        break
            else:
                # opening new position
                trade = self.trade_manager.open_trade(
                    entry_ts=ts,
                    entry_price=current_price,
                    exposure=target_pos,  # actual position size with sign
                    ticker=ticker
                )
                logger.info(f"  → Opened {ticker} trade | Entry: ${current_price:.2f} | Size: {target_pos:.2f}")

        # log operations
        self.operations.extend(ops)

        return ops if ops else None
    

    def get_unrealized_pnl(self, point):

        tickers = {self.pair[0] : point[1][self.pair[0]],
                   self.pair[1] : point[1][self.pair[1]]}
        
        pnl = self.trade_manager.compute_unrealized_pnl(tickers)

        return pnl


    def get_summary(self):
        """
        print summary statistics of signals
        """
        if not self.signals:
            logger.warning("no signals generated")
            return

        logger.info(f"\n{'='*100}")
        logger.info(f"  STRATEGY SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"  Total signals: {len(self.signals)}")

        long_count = sum(1 for s in self.signals if s.exposure == 1)
        short_count = sum(1 for s in self.signals if s.exposure == -1)
        close_count = sum(1 for s in self.signals if s.exposure == 0)

        logger.info(f"  Long entries: {long_count}")
        logger.info(f"  Short entries: {short_count}")
        logger.info(f"  Closes: {close_count}")

        logger.info(f"\n  Total operations: {len(self.operations)}")

        # Get current positions from TradeManager
        current_positions = {}
        for ticker in self.pair:
            pos = self._get_current_position(ticker)
            if abs(pos) > 1e-9:
                current_positions[ticker] = pos
        logger.info(f"  Current positions: {current_positions if current_positions else 'None'}")

        # PnL statistics from TradeManager
        completed_trades = self.trade_manager.complete_trades
        if completed_trades:
            logger.info(f"\n  PnL Statistics:")
            logger.info(f"    Completed trades: {len(completed_trades)}")

            pnls = [t.pnl for t in completed_trades]
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)

            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]

            win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0

            if total_pnl > 0:
                logger.success(f"    Total PnL:     ${total_pnl:8.2f} ✓")
            elif total_pnl < 0:
                logger.error(f"    Total PnL:     ${total_pnl:8.2f} ✗")
            else:
                logger.info(f"    Total PnL:     ${total_pnl:8.2f}")

            logger.info(f"    Average PnL:   ${avg_pnl:8.2f}")
            logger.info(f"    Best trade:    ${max(pnls):8.2f}")
            logger.info(f"    Worst trade:   ${min(pnls):8.2f}")
            logger.info(f"    Win rate:      {win_rate:7.1f}%")
            logger.info(f"    Winners:       {len(winning_trades)}")
            logger.info(f"    Losers:        {len(losing_trades)}")
        else:
            logger.warning(f"\n  No completed trades yet")

        # display info about current open trades
        if self.trade_manager.current_trades:
            logger.info(f"\n  Open trades: {len(self.trade_manager.current_trades)}")

        logger.info(f"{'='*50}\n")

        self.plot_cumulative_pnl()

    def plot_cumulative_pnl(self):
        """Plot cumulative PnL over time with blue line and colored fill (green > 0, red < 0)."""
        completed_trades = self.trade_manager.complete_trades

        if not completed_trades:
            logger.warning("No completed trades to plot")
            return

        # Get PnL series sorted by exit time
        pnl_series = sorted(
            [(trade.exit_ts, trade.pnl) for trade in completed_trades],
            key=lambda x: x[0]
        )

        timestamps = [ts for ts, _ in pnl_series]
        pnls = [pnl for _, pnl in pnl_series]

        # Calculate cumulative PnL
        cumulative_pnl = np.cumsum(pnls)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot blue line for cumulative PnL
        ax.plot(timestamps, cumulative_pnl, color='green', label='Cumulative PnL')

        # Fill between line and zero with conditional coloring
        ax.fill_between(
            timestamps,
            0,
            cumulative_pnl,
            where=(cumulative_pnl >= 0),
            color='green',
            alpha=0.3,
            interpolate=True,
            label='Profit'
        )
        ax.fill_between(
            timestamps,
            0,
            cumulative_pnl,
            where=(cumulative_pnl < 0),
            color='red',
            alpha=0.3,
            interpolate=True,
            label='Loss'
        )

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative PnL ($)', fontsize=12)
        ax.set_title(f'Cumulative PnL - {self.pair[0]}/{self.pair[1]} Spread', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Add final PnL annotation
        final_pnl = cumulative_pnl[-1]
        final_color = 'green' if final_pnl >= 0 else 'red'
        ax.text(
            0.02, 0.98,
            f'Final PnL: ${final_pnl:.2f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=final_color, alpha=0.3)
        )

        plt.tight_layout()
        plt.show(block=False)

    def plot_strategy_analysis(self):
        """plot strategy analysis: prices, spread, and signals."""
        if not self.price_history:
            logger.warning("No price history to plot")
            return

        # Convert price history to arrays
        timestamps = [p[0] for p in self.price_history]
        s1_prices = [p[1] for p in self.price_history]
        s2_prices = [p[2] for p in self.price_history]
        spreads = [p[3] for p in self.price_history]
        z_scores = [p[4] if p[4] is not None else np.nan for p in self.price_history]

        # Get entry and exit signals
        entry_long = [(s.ts, s.z) for s in self.signals if s.exposure == 1]
        entry_short = [(s.ts, s.z) for s in self.signals if s.exposure == -1]
        exits = [(s.ts, s.z) for s in self.signals if s.exposure == 0]

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Stock Prices
        ax1.plot(timestamps, s1_prices, label=self.pair[0], color='blue', linewidth=1.5)
        ax1.plot(timestamps, s2_prices, label=self.pair[1], color='orange', linewidth=1.5)

        # Mark entries/exits on price chart
        for ts, _ in entry_long:
            ax1.axvline(x=ts, color='green', alpha=0.6, linestyle='--', linewidth=1)
        for ts, _ in entry_short:
            ax1.axvline(x=ts, color='red', alpha=0.6, linestyle='--', linewidth=1)
        for ts, _ in exits:
            ax1.axvline(x=ts, color='gray', alpha=0.6, linestyle=':', linewidth=1)

        ax1.set_ylabel('Price ($)', fontsize=11)
        ax1.set_title(f'Stock Prices - {self.pair[0]} / {self.pair[1]}', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Spread
        ax2.plot(timestamps, spreads, label='Spread', color='purple', linewidth=1.5)

        # Mark entries/exits on spread
        for ts, _ in entry_long:
            ax2.axvline(x=ts, color='green', alpha=0.6, linestyle='--', linewidth=1)
        for ts, _ in entry_short:
            ax2.axvline(x=ts, color='red', alpha=0.6, linestyle='--', linewidth=1)
        for ts, _ in exits:
            ax2.axvline(x=ts, color='gray', alpha=0.6, linestyle=':', linewidth=1)

        # Add mean line
        mean_spread = np.nanmean(spreads)
        ax2.axhline(y=mean_spread, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Mean')

        ax2.set_ylabel('Spread', fontsize=11)
        ax2.set_title('Spread', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Z-Score
        ax3.plot(timestamps, z_scores, label='Z-Score', color='darkblue', linewidth=1.5)

        # Add threshold lines
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax3.axhline(y=self.strategy_config['z_entry'], color='green', linestyle='--', linewidth=1, alpha=0.7, label=f'Entry Threshold ({self.strategy_config["z_entry"]})')
        ax3.axhline(y=self.strategy_config['z_exit'], color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Exit Threshold ({self.strategy_config["z_exit"]})')

        # Mark signals with scatter points
        if entry_long:
            entry_long_ts = [e[0] for e in entry_long]
            entry_long_z = [e[1] for e in entry_long]
            ax3.scatter(entry_long_ts, entry_long_z, color='green', marker='^', s=100, zorder=5, label='Long Entry')

        if entry_short:
            entry_short_ts = [e[0] for e in entry_short]
            entry_short_z = [e[1] for e in entry_short]
            ax3.scatter(entry_short_ts, entry_short_z, color='red', marker='v', s=100, zorder=5, label='Short Entry')

        if exits:
            exit_ts = [e[0] for e in exits]
            exit_z = [e[1] for e in exits]
            ax3.scatter(exit_ts, exit_z, color='black', marker='x', s=100, zorder=5, label='Exit')

        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Z-Score', fontsize=11)
        ax3.set_title('Z-Score with Entry/Exit Signals', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)



# pair to trade
pair = ["ORCL", "AMD"]

start = "2025-10-20"
bar = "1d"

# strategy configuration
config = {
    "z_entry": -1.5,
    "z_exit": 1.5,
    "maxhold": 100,
    "win_len":200,
}

# create manager
manager = CPMRManager(pair=pair, strategy_config=config, base_units=1.0)

# check health and visualize
logger.info("=== HEALTH CHECK ===")
health = manager.check_health(fit_start=start, fit_bar=bar, show_plot=True)

# initialize with historical data
manager.initialize(fit_start=start, fit_bar=bar)

# get test data as iterator
logger.info("\n=== BACKTESTING ===")
logger.info("loading test data...")

# data_iter = load_data(
#     tickers=pair,
#     start="2025-01-01",
#     bar="1h",
#     as_iter=True
# )

data_iter = load_data(
    tickers=pair,
    start=start,
    bar=bar,
    as_iter=True
)

# process each point through manager
logger.info("processing data points...")
for point in data_iter:
    ops = manager.process_point(point)

    pnl = manager.get_unrealized_pnl(point)

    if pnl != 0:
        logger.info(f"Unrealized PnL: {pnl}")

    time.sleep(0.01)

# show summary
manager.get_summary()

# plot strategy analysis
manager.plot_strategy_analysis()


input("")
