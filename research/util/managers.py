from research.util.Interfaces import StockOperation, MarketPointData, Trade
import uuid

class TradeManager():
    def __init__(self):
        self.complete_trades = []
        self.current_trades = []  # {"entry_ts", "entry_price", "exposure", "entry_z"}

    def open_trade(self, entry_ts, entry_price, exposure, ticker):
        trade = Trade(
            uuid=uuid.uuid4(),
            entry_ts=entry_ts,
            entry_price=entry_price,
            exit_ts=None,
            exposure=exposure,
            pnl=None,
            ticker=ticker
        )
        self.current_trades.append(trade)
        return trade
    
    def _get_trade_by_uuid(self, uuid):
        return next((trade for trade in self.current_trades if trade.uuid == uuid), None)
    
    def close_trade(self, trade_uuid, market_point: MarketPointData):
        """
        Close a trade and track realized PnL.

        Args:
            trade_uuid: UUID of the trade to close
            market_point: Market data point containing exit timestamp and price
        """
        # Get the trade from current trades
        trade = self._get_trade_by_uuid(trade_uuid)

        if trade is None:
            raise ValueError(f"Trade with UUID {trade_uuid} not found in current trades")

        # Calculate realized PnL
        pnl = self._calculate_pnl(trade=trade, current_price=market_point.price)

        # Update trade with exit information
        trade.exit_ts = market_point.ts
        trade.pnl = pnl

        # Remove from current trades and add to completed trades
        self.current_trades.remove(trade)
        self.complete_trades.append(trade)

        return trade

    def get_realized_pnl_series(self):
        """
        get realized PnL data with timestamps for plotting.

        Returns:
            List of dicts with 'ts' (exit timestamp) and 'pnl' (realized PnL)
        """
        return [
            {"ts": trade.exit_ts, "pnl": trade.pnl}
            for trade in self.complete_trades
            if trade.exit_ts is not None and trade.pnl is not None
        ]

    def _calculate_pnl(self, trade, current_price) -> float:
        """
        calculate unrealized PnL for a given trade based on current price.

        Args:
            trade_uuid: UUID of the trade to calculate PnL for
            current_price: Current market price

        Returns:
            PnL value (positive = profit, negative = loss)
        """
        # trade = self._get_trade_by_uuid(trade_uuid)

        # if trade is None:
        #     raise ValueError(f"Trade with UUID {trade_uuid} not found")


        # long (exposure > 0): profit when price increases
        # short (exposure < 0): profit when price decreases
        pnl = (current_price - trade.entry_price) * trade.exposure

        return pnl
    
    def request_tickers_for_current_trades(self):
        """
        Get list of unique tickers for all current open trades.

        Returns:
            List of ticker strings
        """
        tickers = [trade.ticker for trade in self.current_trades if trade.ticker is not None]
        # Return unique tickers only
        return list(set(tickers))

    def compute_unrealized_pnl(self, market_data: dict) -> float:
        """
        Calculate total unrealized PnL for all current trades.

        Args:
            market_data: Dictionary mapping ticker -> price (float) or ticker -> MarketPointData
                        Example: {'AAPL': 150.0, 'MSFT': 300.0}
                        or {'AAPL': MarketPointData(...), 'MSFT': MarketPointData(...)}

        Returns:
            Total unrealized PnL across all open trades
        """
        total_pnl = 0.0

        for trade in self.current_trades:
            if trade.ticker not in market_data:
                raise ValueError(f"No market data provided for ticker {trade.ticker}")

            # Handle both dict[ticker, price] and dict[ticker, MarketPointData]
            market_value = market_data[trade.ticker]
            
            current_price = market_value

            # Calculate PnL for this trade
            pnl = self._calculate_pnl(trade=trade, current_price=current_price)
            total_pnl += pnl

        return total_pnl

class OperationManager:
    def __init__(self):
        self.operations = []
        self.signals = []
    
    def add_signal(self, signal):
        self.signals.append(signal)
    
    def add_operation(self, operation: StockOperation):
        self.operations.append(operation)