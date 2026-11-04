import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from features import create_features
from targets import create_target
from sklearn.metrics import accuracy_score, classification_report

saved = joblib.load("models/random_forest_model_final.joblib")
model = saved["model"]
feature_columns = saved["features"]

part3_availabe = Path('data/PART3/01.csv').exists()
parts = ["PART1", 'PART2', "PART3"] if part3_availabe else ["PART1", "PART2"]

all_predictions = []
all_actual = []

for i in range(1,11):
    stock_id = f"{i:02d}"
    datasets = []

    for part in parts:
        df = pd.read_csv(f"data/{part}/{stock_id}.csv")
        df = df.rename(columns={"Index": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        datasets.append(df)

    #include earlier history when calculating indicators
    eval_start = sum(len(df) for df in datasets[:-1])
    combined = pd.concat(datasets, ignore_index=True)

    combined = create_features(combined)
    combined = create_target(combined)
    combined = combined.replace([np.inf, -np.inf], np.nan)

    eval = combined.iloc[eval_start - 1:-1].copy()
    eval = eval.dropna(subset=feature_columns + ["Target"])

    predictions = model.predict(eval[feature_columns])

    if part3_availabe:
        all_predictions.extend(predictions)
        all_actual.extend(eval["Target"].astype(int))

    #predict the movement after the latest avaiable day
    latest = combined[feature_columns].iloc[[-1]]
    next_prediction = model.predict(latest)[0]

    print(f"\nStock {stock_id}")
    print("Latest date:", combined["Date"].iloc[-1].date())
    print("Predicted next movement", 'UP' if next_prediction == 1 else "down or not changed")



if part3_availabe:
    print("\nUnseen data evaluation results")
    print("Accuracy:", accuracy_score(all_actual, all_predictions))
    print(classification_report(all_actual,all_predictions, zero_division=0))
