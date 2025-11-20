from typing import List, Union, Iterable, Optional, Iterator, Tuple

import pandas as pd
import yfinance as yf
import datetime


def get_snp_tickers():
    snp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    tables = pd.read_html(
        snp_url,
        match="Symbol",
        storage_options={"User-Agent": "Mozilla/5.0"}
    )

    sp500 = tables[0]

    tickers = sp500["Symbol"].tolist()
    return tickers

def today_str(minus: int = 0, fmt: str = "%Y-%m-%d") -> str:
    """Return today's date string in given timezone, shifted back by `minus` days."""
    now = pd.Timestamp.now() - pd.Timedelta(days=minus)  # minus is days
    return now.strftime(fmt)


def load_data(
    tickers: Union[str, Iterable[str]],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    bar: str = "1d",
    as_iter: bool = False,
    *,
    auto_adjust: bool = True,
) -> Union[pd.DataFrame, Iterator[Tuple[pd.Timestamp, pd.Series]]]:
    if end is None:
        end = today_str()

    if isinstance(tickers, str):
        symbols: List[str] = [tickers]
        ticker_arg = tickers
    else:
        symbols = list(tickers)
        ticker_arg = " ".join(symbols)


    data = yf.download(
        tickers=ticker_arg,
        start=start,
        end=end,
        interval=bar,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=True,
        threads=True,
    )

    if not as_iter:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        df = data
    else:
        sym = symbols[0] if symbols else "TICKER"
        df = data.copy()
        df.columns = pd.MultiIndex.from_product([[sym], df.columns])

    def _iter() -> Iterator[Tuple[pd.Timestamp, pd.Series]]:
        for ts in df.index:
            yield ts, df.loc[ts]

    return _iter()

def load_data_offline(
        tickers: Union[str, Iterable[str]],
        start: str = "2020-01-01",
        end: Optional[str] = None,
        bar: str = "1d",
        as_iter = False
    ):

    # # defaults

    # dstart = "2020-01-01"
    # dbar = "1d"

    # start_date = datetime.date(2020, 1, 1)

    # added = start_date + datetime.timedelta(1)

    data = pd.read_csv("research/mean_revision/snp_data")

    # for idx in data()

    data = data[tickers + ["Date"]]

    data = data[data["Date"] >= start]

    if not as_iter:
        return data

    def _iter() -> Iterator[Tuple[pd.Timestamp, pd.Series]]:
        for ts in data.index:
            yield ts, data.loc[ts]

    return _iter()




# data = load_data_offline(tickers=["MAR", "PANW"], start="2020-01-01", as_iter=True)


# print(data)