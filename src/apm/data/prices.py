import yfinance as yf
import pandas as pd


# 定义 MMI 成分股及其股票代码
MMI_tickers = {
    'American Express': 'AXP',
    'AT&T': 'T',
    'Chevron': 'CVX',
    'Coca Cola': 'KO',
    'Disney': 'DIS',
    'Dow': 'DOW',
    'Dupont': 'DD',
    'Eastman Kodak': 'KODK',
    'Exxon': 'XOM',
    'General Electric': 'GE',
    'General Motors': 'GM',
    'IBM': 'IBM',
    'International Paper': 'IP',
    'Johnson & Johnson': 'JNJ',
    'McDonalds': 'MCD',
    'Merck': 'MRK',
    '3M': 'MMM',
    'Philip Morris': 'PM',
    'Procter & Gamble': 'PG',
}

def load_mmi_data(
    tickers=MMI_tickers,
    period='10y',
    interval='1d',
    quote='Close',
):
    """Download price data for MMI index"""
    symbols = list(tickers.values())
    data: pd.DataFrame = yf.download(
        symbols, period=period, interval=interval,
    )[quote].dropna()
    return data


def load_ng_data(
    symbol='NG=F',
    period='10y',
    interval='1d',
):
    """Download price data for Natural Gas Futures"""
    data: pd.DataFrame = yf.download(
        symbol, period=period, interval=interval,
        group_by='ticker',
    )[symbol]
    return data
