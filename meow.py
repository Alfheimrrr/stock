import pandas as pd
import matplotlib.pyplot as plt
from features import create_features
import numpy as np
from targets import create_target
from prepare_data import prepare_data

stocks = {}

for i in range(1,11):
    stockid = f"{i:02d}"

    df1 = pd.read_csv(f"data/PART1/{stockid}.csv")
    df2 = pd.read_csv(f"data/PART2/{stockid}.csv")

    df1 = df1.rename(columns = {"Index": "Date"})
    df2 = df2.rename(columns = {"Index": "Date"})

    for df in [df1,df2]:
        df["Date"] = pd.to_datetime(df["Date"])

    df1 = df1.sort_values("Date").reset_index(drop=True)
    df2 = df2.sort_values("Date").reset_index(drop=True)

    stocks[stockid] = {
        "train" : df1,
        "val" : df2
    }


for stock_id, datasets in stocks.items():

    train = datasets['train']
    val = datasets['val']

    combined = pd.concat(
        [train,val],
        ignore_index=True
    )
    #calculate financial indicators
    combined = create_features(combined)
    #calculate next day prediction target
    combined = create_target(combined)

    stocks[stock_id]['features'] = combined

for stock_id, datasets in stocks.items():
    df = datasets["features"]

    x_train, x_val, y_train, y_val = prepare_data(df)

    stocks[stock_id]["x_train"] = x_train
    stocks[stock_id]["x_val"] = x_val
    stocks[stock_id]["y_train"] = y_train
    stocks[stock_id]["y_val"] = y_val

    

