"""
base classes for unified strategy architecture
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from research.util.Interfaces import MarketPointData

import pandas as pd


class BaseStrategy(ABC):
    """
    base class for all trading strategies LOGIC
    every strategy must inherit from this class
    this should be as abstract as possible
    all the interaction with world should be handled using StrategyManager
    """

    def __init__(self, name: str, config):
        """
        initialize strategy with name and configuration

        args:
            name: unique strategy name
            config: strategy parameters (instruments, thresholds, etc.)
        """
        self.name = name
        self.config = config
        self._initialized = False

    @abstractmethod
    def process(self, market_data: MarketPointData) -> Optional[Any]:
        """
        main method that processes market data and generates signals

        args:
            market_data: current market data (can be dict, dataframe, etc.)

        returns:
            signal object if strategy generates signal, None otherwise
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        initialize strategy (load history, warm up rolling windows, etc.)
        called once before first process() call
        """

class BaseStrategyManager(ABC):
    """
    This class takes a strategy as argument and works as wraper for differetn applications
    """
    def __init__(self, strategy):
        super().__init__()
        self.strat = strategy

    def process_point(self, market_data):
        """
        endpoint that wraps the self.strat.process() method and transforms result to suitable format required by the outer interface
        """