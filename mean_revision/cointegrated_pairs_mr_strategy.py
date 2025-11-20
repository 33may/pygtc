import uuid
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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

from research.util.Interfaces import StockOperation
from research.util.data import load_data, today_str
from research.util.ploting import plot_spread

# --- O(1) rolling stats for fixed-size window ---
class RollingWindowStats:
    """Fixed-size rolling mean/std with O(1) update using sum and sumsq."""

    def __init__(self, win: int):
        assert win > 1, "win must be > 1"
        self.win = win
        self.buf = deque()
        self.sum = 0.0
        self.sumsq = 0.0

    def _pop_oldest(self):
        """Remove oldest value when buffer is full."""
        old = self.buf.popleft()
        self.sum -= old
        self.sumsq -= old * old

    def push(self, x: float) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Add new value. Returns (mean, std, filled)
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
        """Fill buffer from iterable up to win (last win elements are used)."""
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
        """Return (mean, std) for current window if ready, else (None, None)."""
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


def regression(self, s1_data, s2_data):
    X = s1_data.values
    y = s2_data.values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    alpha, beta = model.params
    residuals = y - (alpha + beta * X[:, 1])

    if np.isnan(residuals).any():
        return np.nan, np.nan, float(alpha), float(beta)

    adf_res = adfuller(residuals)
    return adf_res[0], adf_res[1], float(alpha), float(beta)

def compute_spread(s1, s2, alpha, beta):
    return s1 - (alpha + beta * s2)

class StrategyManager:
    """
    receives Target and converts it to concrete market operations
    """
    def __init__(self, s1_ticker: str, s2_ticker: str, *, base_units: float = 1.0, name: str = "pairs_mr"):
        self.s1 = s1_ticker
        self.s2 = s2_ticker
        self.base_units = float(base_units)
        self.name = name

        self.positions: Dict[str, float] = {self.s1: 0.0, self.s2: 0.0}
        self.log: List[Dict] = []
        self.last_exposure: Optional[int] = None

    def process_target(self, target: Target) -> List[StockOperation]:
        """
        convert Target(exposure in {-1,0,+1}) to BUY/SELL ops and update positions.
          u1 = base_units
          u2 = beta * u1
        """
        ts = target.ts

        u1 = float(self.base_units)
        u2 = float(target.beta) * u1

        if target.exposure > 0:
            desired = {self.s1: +u1, self.s2: -u2}  # long spread
        elif target.exposure < 0:
            desired = {self.s1: -u1, self.s2: +u2}  # short spread
        else:
            desired = {self.s1: 0.0, self.s2: 0.0}   # flatten

        # translate deltas to orders
        eps = 1e-12
        ops: List[StockOperation] = []
        for ticker in (self.s1, self.s2):
            cur = float(self.positions.get(ticker, 0.0))
            des = float(desired[ticker])
            delta = des - cur
            if abs(delta) < eps:
                continue
            side = +1 if delta > 0 else -1
            amt = abs(delta)
            ops.append(StockOperation(
                id=uuid.uuid4(),
                ts=ts,
                ticker=ticker,
                exposure=side,
                amount=amt
            ))

        self._apply_fills(ops)
        self._log_event(ts, target, ops)

        self.last_exposure = int(target.exposure)
        return ops

    def _apply_fills(self, ops: List[StockOperation]) -> None:
        for op in ops:
            q = float(self.positions.get(op.ticker, 0.0))
            if op.exposure == +1:
                q += float(op.amount)
            elif op.exposure == -1:
                q -= float(op.amount)
            self.positions[op.ticker] = q

    def _log_event(self, ts: pd.Timestamp, target: Target, ops: List[StockOperation]) -> None:
        """append a compact log entry."""
        event = self._classify_transition(target.exposure)
        self.log.append({
            "ts": ts,
            "event": event,
            "z": float(target.z),
            "beta": float(target.beta),
            "exposure": int(target.exposure),
            "ops": [(op.ticker, op.exposure, op.amount) for op in ops],
            "pos_after": dict(self.positions),
        })

    def _classify_transition(self, new_exp: int) -> str:
        if new_exp == 0:
            return "close"
        if self.last_exposure is None:
            return "open"
        if (new_exp > 0) != (self.last_exposure > 0):
            return "flip"
        return "rebalance"

    def snapshot(self) -> Dict:
        return {"positions": dict(self.positions), "last_exposure": self.last_exposure, "name": self.name}

    def reset(self) -> None:
        self.positions = {self.s1: 0, self.s2: 0}
        self.last_exposure = None
        self.log.clear()


class CointegratedPairsMRStrategy:
    def __init__(self, pair: List[str], z_entry: float, z_exit: float, z_bot_stop: float, z_top_stop: float,
                 max_hold: int, win_len: int, mode: str, verbose: bool = True):
        self.mode = mode

        self.pair = pair

        self.s1 = pair[0]
        self.s2 = pair[1]

        self.z_entry = z_entry
        self.z_exit = z_exit
        self.max_hold = max_hold
        self.z_bot_stop = z_bot_stop
        self.z_top_stop = z_top_stop
        self.win_len = win_len

        self.state = RollingState(rolling_mean=0.0, rolling_std=0.0)
        self._roll = RollingWindowStats(win_len)

        self.alpha = None
        self.beta = None

        self.active_position = None
        self.verbose = verbose

    def check_status(self, start: str, bar: str):
        data = load_data(tickers=self.pair, start=start, bar=bar)["Close"]

        s1_data = data[self.s1]
        s2_data = data[self.s2]

        adf_stat, p_val, self.alpha, self.beta = regression(self, s1_data, s2_data)

        if p_val < 0.05:
            logger.success(f"the pair {self.s1}/{self.s2} is cointegrated. (p_val = {p_val})")
        else:
            logger.warning(f"the pair {self.s1}/{self.s2} failed cointegration test. (p_val = {p_val})")

        plot_spread(s1_data, s2_data, spread=None, alpha=self.alpha, beta=self.beta, stock1_name=self.s1,
                    stock2_name=self.s2)

    def load_history(self, mode) -> None:
        """
        Warm up rolling window using last win_len spreads from provided data.
        - data: DataFrame with columns [self.s1, self.s2] (aligned by index).
        - recompute_alpha_beta: if True, compute alpha/beta on full provided history.
        """
        # ensure we have necessary columns

        data = yf.download(self.pair, start=today_str(minus=self.win_len * 2))["Close"]

        s1 = data[self.s1].astype(float)
        s2 = data[self.s2].astype(float)
        s1, s2 = s1.align(s2, join="inner")

        # build full spread series and warm up last win points
        spread_series = compute_spread(s1, s2, self.alpha, self.beta)
        self._roll.warmup(spread_series)

        # update state
        mean, std = self._roll.current()
        self.state.rolling_mean = 0.0 if mean is None else float(mean)
        self.state.rolling_std = 0.0 if std is None else float(std)

    def process(self, point) -> Target | None:
        if not self._roll.ready():
            self.load_history(mode=self.mode)

        ts, point = point
        point = point["Close"]

        data_s1 = float(point[self.s1])
        data_s2 = float(point[self.s2])

        spread = compute_spread(data_s1, data_s2, self.alpha, self.beta)

        mean, std, filled = self._roll.push(spread)
        if not filled or mean is None or std is None or std == 0.0:
            return None

        z_score = (spread - mean) / std

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




pair = ["MAR", "PANW"]

strat = CointegratedPairsMRStrategy(
    pair=pair, z_entry=-1.5, z_exit=1.5,
    z_bot_stop=-4, z_top_stop=4, max_hold=100,
    win_len=30, mode="1d"
)

strat.check_status(start="2020-01-01", bar="1d")
strat.load_history(mode="1d")

manager = StrategyManager(s1_ticker=pair[0], s2_ticker=pair[1], base_units=1, name="pairs_mr")

data_runner = load_data(tickers=pair, start="2024-01-01", as_iter=True)

def expo_to_str(x: int) -> str:
    return "BUY" if x == 1 else "SELL" if x == -1 else "CLOSE"

for item in data_runner:
    target = strat.process(item)
    if not target:
        continue

    # produce concrete market operations and update positions
    ops = manager.process_target(target)

    # print executed operations
    for op in ops:
        print(f"{op.ts} | {op.ticker:>5} | {expo_to_str(op.exposure):>4} | amount={op.amount}")