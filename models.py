from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

#logistic regression with features
def train_logistic_regression(x_train, y_train):

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(x_train, y_train)

    return model
#random forest using 100 trees with a max depth of 5
def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(
        max_depth=5,
        n_estimators = 100,
        random_state=63,
        n_jobs=1
    )
    model.fit(x_train, y_train)

    return model
#XGboost
def train_xgboost(x_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=89,
        eval_metric="logloss"
    )
    model.fit(x_train,y_train)

    return model
#SVM with rbf kernel
def train_svm(x_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0)
    )
    model.fit(x_train,y_train)

    return model



