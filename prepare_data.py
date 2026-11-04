import numpy as np

def prepare_data(df):
    df = df.copy()
    #split data 80% training 20% validation
    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index].copy()
    val = df.iloc[split_index:].copy()
    #remove last training target
    train.loc[train.index[-1], "Target"] = np.nan

    #remove missing and infinite values
    train = train.replace([np.inf, -np.inf], np.nan).dropna()
    val = val.replace([np.inf, -np.inf], np.nan).dropna()

    #select input features and target values
    feature_columns = [
        col for col in df.columns
        if col not in ["Date", 'Target']
    ]
    x_train = train[feature_columns]
    y_train = train["Target"].astype(int)

    x_val = val[feature_columns]
    y_val = val["Target"].astype(int)

    return x_train,x_val,y_train,y_val