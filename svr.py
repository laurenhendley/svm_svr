#### imports

# Getting the dataset
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def model_generation(train_X, test_X, train_Y, test_Y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel = "rbf"))
    ])

    model.fit(train_X, train_Y)
    y_pred = model.predict(test_X)

    return y_pred, model

def accuracy(y_test, y_pred):
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("R^2: ", r2_score(y_test, y_pred))


if __name__ == "__main__":
    ds = datasets.load_diabetes()

    train_X, test_X, train_Y, test_Y = train_test_split(ds.data, ds.target, test_size = 0.3, random_state = 42)

    Y_pred, model = model_generation(train_X, test_X, train_Y, test_Y)

    accuracy(test_Y, Y_pred)


