import joblib
import pandas as pd
from features import create_features

#load final trained random forest
saved = joblib.load("random_forest_model_final.joblib")
model = saved["model"]
feature_columns = saved["features"]

df = pd.read_csv("data/PART2/01.csv")

df = df.rename(columns ={"Index": "Date"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
#calculate financial indicators
df = create_features(df)

#use latest day to predict the next movement
latest = df[feature_columns].iloc[[-1]]

if latest.isna().any().any():
    raise ValueError("Latest row contains missing feature")

prediction = model.predict(latest)[0]

print("Stock: 01")
print("Latest date:", df["Date"].iloc[-1])
print("Predicted next day movement:",
      "Up" if prediction == 1 else "down or not changed")