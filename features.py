import pandas as pd

def create_features(df):
    df = df.copy()
    # daily returns
    df["Return"] = df["Close"].pct_change()

    #returns from previous trading days
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)

    #simple moving averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    #exponential moving averages
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    #relative strength index
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(alpha = 1/14, adjust=False, min_periods=14).mean()
    average_loss = loss.ewm(alpha = 1/14, adjust=False, min_periods=14).mean()

    rs = average_gain / average_loss

    df["RSI_14"] = 100 - (100 / (1 + rs))

    #MACD and signal line
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    df["MACD_signal"] = df["MACD"].ewm(span = 9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_signal"]
    #Volatility over 20 days of trading
    df["Volatility_20"] = df["Return"].rolling(window=20).std()

    #percentage change in trading volume
    df["Volume_change"] = df["Volume"].pct_change()
    # daily price range relative to closing price
    df["High_low_range"] = (
        (df['High'] - df['Low']) / df['Close']
         
    )
    return df