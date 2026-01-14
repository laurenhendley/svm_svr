#### imports

# Getting the dataset
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier


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
    base = svm.SVC(kernel = 'linear')
    # Using bagging to make it more accurate
    classifier = BaggingClassifier(estimator = base)
    classifier.fit(train_X, train_Y)

    y_pred = classifier.predict(test_X)

    return y_pred, classifier



def model_evaluation(Y_test, Y_pred):
    print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))



if __name__ == "__main__":
    ds = datasets.load_breast_cancer()

    train_X, test_X, train_Y, test_Y = train_test_split(ds.data, ds.target, test_size = 0.3, random_state = 109)

    Y_pred, classifier = model_generation(test_X,test_Y,train_X,train_Y)

    model_evaluation(test_Y, Y_pred)


