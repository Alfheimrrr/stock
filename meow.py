import pandas as pd
import matplotlib.pyplot as plt
from features import create_features
import numpy as np
from targets import create_target
from prepare_data import prepare_data
from models import train_logistic_regression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier

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


entire_x_train = []
entire_y_train = []
entire_x_val = []
entire_y_val = []


for stock_id, datasets in stocks.items():
    train = datasets ["train"]
    val = datasets['val']

    if train["Date"].max() >= val["Date"].min():
        raise ValueError(
            f"Stock {stock_id}: part 1 and part 2 overlap or out of order"
        )
    

    combined = pd.concat(
        [train,val],
        ignore_index=True
    )

    combined = create_features(combined)
    combined = create_target(combined)

    x_train, x_val, y_train, y_val = prepare_data(combined)

    entire_x_train.append(x_train)
    entire_y_train.append(y_train)

    entire_x_val.append(x_val)
    entire_y_val.append(y_val)
    

x_train = pd.concat(entire_x_train, ignore_index=True)
y_train = pd.concat(entire_y_train, ignore_index=True)

x_val = pd.concat(entire_x_val, ignore_index=True)
y_val = pd.concat(entire_y_val, ignore_index=True)

print("combined training observations:", len(x_train))

print("combined val observations:", len(x_val))

model = train_logistic_regression(x_train, y_train)

predictions = model.predict(x_val)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(x_train, y_train)

baseline_predictions = baseline.predict(x_val)

print("\nCombined logistic regression results:")

print(classification_report(
    y_val,
    predictions,
    zero_division=0
))
print(
    "Logistic regression accuracy:",
    accuracy_score(y_val,predictions)
)

print(
    'Baseline accuracy:',
    accuracy_score(y_val,baseline_predictions)
)

