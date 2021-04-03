import requests
import pandas as pd


class DataStore:
    store = {}

    @classmethod
    def loadKlineHistory(cls, symbol="dotusdt", timeframe="1m", klines=1000):
        sig = symbol + "@" + timeframe + "-" + str(klines)
        if sig in cls.store:
            return cls.store[sig]
        uri = "https://api.binance.com/api/v3/klines?symbol={}&interval={}&limit={}"
        uri = uri.format(symbol.upper(), timeframe, klines)
        r = requests.get(uri)
        data = r.json()
        cls.store[sig] = data
        return data

    @classmethod
    def loadKlineClosesHistory(cls, symbol="dotusdt", timeframe="1m", klines=1000):
        cols = ["OpenTime",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "CloseTime",
                "Quote-asset-volume",
                "Number-of-trades",
                "Taker-buy-base-asset-volume",
                "Taker-buy-quote-asset-volumze",
                "Ignore"]
        hist = cls.loadKlineHistory(symbol, timeframe, klines)
        df = pd.DataFrame(hist, columns=cols)
        df["Close"] = df["Close"].apply(lambda x: float(x))
        data = df["Close"].to_numpy()
        return data
