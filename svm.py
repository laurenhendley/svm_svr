# LAST MEASURED ACCURACY: 97%

#### imports

# Getting the dataset
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier

import matplotlib.pyplot as mpl
import numpy as np



# Testing out aspects of the model
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


# Generating the model
def model_generation(train_X, test_X, train_Y):
    # Use SVM with linear kernel
    base = svm.SVC(kernel = 'linear')

    # Using bagging to make the model more accurate
    classifier = BaggingClassifier(estimator = base)
    classifier.fit(train_X, train_Y)

    # Predict with the model
    y_pred = classifier.predict(test_X)

    return y_pred, classifier


# Getting the accuracy of the model
def model_evaluation(Y_test, Y_pred):
    print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))


# Plotting the results
def plot_metrics(epochs,history):
    N = epochs
    mpl.style.use("ggplot")
    mpl.figure()

    mpl.plot(np.arange(0, N), history.history["loss"], label = "train_loss")
    mpl.plot(np.arange(0, N), history.history["val_loss"], label = "val_loss")
    mpl.plot(np.arange(0, N), history.history["accuracy"], label = "accuracy")
    mpl.plot(np.arange(0, N), history.history["val_accuracy"], label = "val_accuracy")

    mpl.title("Training loss and accuracy on brain tumor dataset")
    mpl.xlabel("Epoch")
    mpl.ylabel("Loss/accuracy")
    mpl.legend(loc="lower left")
    mpl.savefig("brainTumorPlot.jpg")



# Main function
if __name__ == "__main__":
    # Load the dataset
    ds = datasets.load_breast_cancer()

    # Train/test split
    train_X, test_X, train_Y, test_Y = train_test_split(ds.data, ds.target, test_size = 0.3, random_state = 109)

    # Generate model
    Y_pred, classifier = model_generation(train_X, test_X, train_Y)

    # Evaluating the model
    model_evaluation(test_Y, Y_pred)

    # Plotting the accuracy 
    plot_metrics(10, classifier)


