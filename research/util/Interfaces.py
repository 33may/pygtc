import uuid
from dataclasses import dataclass
import numpy as np
from typing import Optional
import pandas as pd


@dataclass
class StockOperation:
    id: uuid.UUID
    ts: pd.Timestamp
    ticker: str
    exposure: int # -1 (sell), +1 (buy), 0 (close)
    amount: float


@dataclass
class MarketPointData:
    """
    single market data point for strategy processing
    strategy should work only with this class, isolated from external data sources
    """
    ts: pd.Timestamp
    # prices array for multiple instruments [price1, price2, ...]
    price: float


@dataclass
class Trade:
    """
    trade object used for internal trades representation, TODO in future we should haev unified internal class and wrappers for different markets trade entities (eg IBKR sends trade info -> convert to Trade -> use inside system)
    """
    uuid: uuid.UUID
    entry_ts: pd.Timestamp
    entry_price: float
    exit_ts: Optional[pd.Timestamp]
    exposure: int
    pnl: Optional[float]
    ticker: Optional[str]