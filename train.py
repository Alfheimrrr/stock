import numpy as np
import pandas as pd
from features import create_features
from targets import create_target
from prepare_data import prepare_data
from models import train_logistic_regression, train_random_forest, train_xgboost, train_svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
import joblib


#load both parts of data for each stock
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

#store the training and validation data for all ten stocks
entire_x_train = []
entire_y_train = []
entire_x_val = []
entire_y_val = []

final_x = []
final_y =[]

# calculate features and split each stock in date order
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
    #keep all usable rows for the final model
    final_data = combined.replace([np.inf, -np.inf], np.nan).dropna()

    final_x.append(final_data[x_train.columns])
    final_y.append(final_data["Target"].astype(int))
    
#combine data from all ten stocks into one training set
x_train = pd.concat(entire_x_train, ignore_index=True)
y_train = pd.concat(entire_y_train, ignore_index=True)

x_val = pd.concat(entire_x_val, ignore_index=True)
y_val = pd.concat(entire_y_val, ignore_index=True)

print("combined training observations:", len(x_train))

print("combined val observations:", len(x_val))

#train and compare different models on the data
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
rf_model = train_random_forest(x_train, y_train)
rf_train_predictions = rf_model.predict(x_train)

print("Random forest training accuracy", accuracy_score(y_train, rf_train_predictions))

rf_predictions = rf_model.predict(x_val)
print("\nRandom Forest results:")

print(classification_report(
    y_val,
    rf_predictions,
    zero_division=0
))
print(
    "Random forest accuracy:",
    accuracy_score(y_val,rf_predictions)
)

xg_model = train_xgboost(x_train, y_train)
xg_model_train_predictions = xg_model.predict(x_train)

print("XGBoost training accuracy", accuracy_score(y_train, xg_model_train_predictions))

xg_predictions = xg_model.predict(x_val)
print("\nXGboost results:")

print(classification_report(
    y_val,
    xg_predictions,
    zero_division=0
))
print(
    "xgboost accuracy:",
    accuracy_score(y_val,xg_predictions)
)

svm_model = train_svm(x_train,y_train)

svm_pred = svm_model.predict(x_val)

print("\nSVM results:")
print(classification_report(y_val, svm_pred))
print("SVM accuracy:", accuracy_score(y_val,svm_pred))
#retrain the random forest on all data
final_x = pd.concat(final_x, ignore_index=True)
final_y = pd.concat(final_y, ignore_index=True)
random_forest_model_final = train_random_forest(final_x, final_y)

# save the final model and its feature
joblib.dump(
    {
        "model": random_forest_model_final,
        "features": final_x.columns.tolist()
    },
    "models/random_forest_model_final.joblib"
)
print("\nRandom Forest with all training data saved")
print("Training observations:", len(final_x))