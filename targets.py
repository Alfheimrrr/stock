import pandas as pd

def create_target(df):

    df = df.copy()
    #get next days closing price
    next_close = df['Close'].shift(-1)
    #if price goes up = 1, else = 0
    df['Target'] = (next_close > df["Close"]).astype(int)

    #remove the target for final day if price unknown
    df.loc[next_close.isna(), 'Target'] = pd.NA

    return df