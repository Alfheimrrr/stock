import pandas as pd

def create_target(df):

    df = df.copy()

    next_close = df['Close'].shift(-1)

    df['Target'] = (next_close > df["Close"]).astype(int)

    df.loc[next_close.isna(), 'Target'] = pd.NA

    return df