""" LAST MEASURED ACCURACY
MSE:  3175.2840124100535
MAE:  42.71147567093662
R^2:  0.4118001135784157
CV R2:  0.45896301212739055
"""


#### imports

# Getting the dataset
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.linear_model import Ridge

import matplotlib.pyplot as mpl
import numpy as np


def model_exploring(ds):
    # Getting the features and labels of the dataset
    print(f"Features: {ds.feature_names}")
    print(f"Labels: {ds.target_names}")

    # Shape of the dataset
    print(ds.data.shape)

    # Getting top 5 records
    print(ds.data[0:5])

    # Target set
    print(ds.target)


def model_generation(train_X, test_X, train_Y):
    param_grid = {
        "svr__C": np.logspace(-1,4,100),
        "svr__gamma": np.logspace(-4,1,10),
        "svr__epsilon": [0.01, 0.1, 0.5, 1.0]
    }

    pipline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel = "rbf"))
    ])

    model = GridSearchCV(
        pipline,
        param_grid,
        scoring="r2",
        cv=5,
        n_jobs=-1,
        refit=True
    )

    train_y_log = np.log1p(train_Y)

    model.fit(train_X, train_y_log)

    best_model = model.best_estimator_

    y_pred_log = best_model.predict(test_X)
    y_pred = np.expm1(y_pred_log)

    return y_pred, best_model

def accuracy(y_test, y_pred):
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("R^2: ", r2_score(y_test, y_pred))


def plot_metrics(model, ds):
    train_size, train_score, val_score = learning_curve (
        model,
        ds.data,
        ds.target,
        cv = 5,
        scoring="r2",
        train_sizes=np.linspace(0.1,1.0,10)
    )

    train_mean = train_score.mean(axis=1)
    val_mean = val_score.mean(axis = 1)

    mpl.figure()
    mpl.plot(train_size,train_mean,label="Training R2")
    mpl.plot(train_size,val_mean,label="Validation R2")
    mpl.xlabel("Training Samples")
    mpl.ylabel("R2 score")
    mpl.title("Diabetes DS SVR LC")
    mpl.legend()
    mpl.show()


if __name__ == "__main__":
    ds = datasets.load_diabetes()

    train_X, test_X, train_Y, test_Y = train_test_split(ds.data, ds.target, test_size = 0.3, random_state = 42)

    Y_pred, model = model_generation(train_X, test_X, train_Y)

    accuracy(test_Y, Y_pred)

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    scores = cross_val_score(
        model,
        ds.data,
        ds.target,
        cv = 10,
        scoring ="r2"
    )

    print("CV R2: ", scores.mean())

    plot_metrics(model, ds)


