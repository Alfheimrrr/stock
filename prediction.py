import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from features import create_features
from targets import create_target
from sklearn.metrics import accuracy_score, classification_report

#load the final model
saved = joblib.load("models/random_forest_model_final.joblib")
model = saved["model"]
feature_columns = saved["features"]

#use part2 unless part3 is present
part3_availabe = Path('data/PART3/01.csv').exists()
parts = ["PART1", 'PART2', "PART3"] if part3_availabe else ["PART1", "PART2"]

all_predictions = []
all_actual = []
trading_balance = []
holdings = []

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

    #include the day before part3 to predict its first movements
    eval = combined.iloc[eval_start - 1:-1].copy()
    eval = eval.dropna(subset=feature_columns + ["Target"])

    predictions = model.predict(eval[feature_columns])
    #trading strategy if part 3 is avaliable
    if part3_availabe:
        all_predictions.extend(predictions)
        all_actual.extend(eval["Target"].astype(int))
        #use next days prices to simulate trade
        prices = combined.loc[eval.index + 1, ["Open", "Close"]].to_numpy()

        cash = 1000.0
        shares = 0.0
        fee = 0.001

        for signal, (open_price,close_price) in zip(predictions, prices):
            if signal == 1 and shares == 0:
                shares = cash * (1-fee) / open_price
                cash = 0.0
            elif signal == 0 and shares > 0:
                cash = shares * open_price * (1-fee)
                shares = 0.0
        #calculate final trading balance
        final_balance = cash + shares * prices[-1, 1] * (1 - fee)
        #compare with buying and holding the stock
        buy_hold = (1000 * (1-fee) / prices[0,0] * prices[-1, 1] * (1 - fee))

        trading_balance.append(final_balance)
        holdings.append(buy_hold)

    #predict the movement after the latest avaiable day
    latest = combined[feature_columns].iloc[[-1]]
    next_prediction = model.predict(latest)[0]

    print(f"\nStock {stock_id}")
    print("Latest date:", combined["Date"].iloc[-1].date())
    print("Predicted next movement", 'UP' if next_prediction == 1 else "down or not changed")


#report performance if part3 is used
if part3_availabe:
    print("\nUnseen data evaluation results")
    print("Accuracy:", accuracy_score(all_actual, all_predictions))
    print(classification_report(all_actual,all_predictions, zero_division=0))

    total = sum(trading_balance)
    hold_total = sum(holdings)

    print("\nTrading strategy results")
    print(f"Starting balance: £10000")
    print(f"Final balance: £{total:.2f}")
    print(f"Profit / loss: £{total - 10000:.2f}")
    print(f"Return: {(total / 10000 -1) * 100:.2f}%")
    print(f"Buy and hold final balance: £{hold_total:.2f}")