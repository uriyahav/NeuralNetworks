from datetime import time
from turtle import st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

import itertools
import warnings
import matplotlib
from jedi.api.refactoring import inline
from numpy.random import seed
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
random_state = 42
warnings.filterwarnings('ignore')
def main():
    #set data
    data = pd.read_csv('wpbc.csv')
    data = data.replace(["N", "R"], [-1, 1])
    labels = data[["b"]]
    data = data.drop(["a","c","f","g","i","p","q","s","x","z","aa"], axis=1)
    data = data.replace(['?'], [6])
    data_train = data[:round(len(data) * 6.6 / 10)]
    data_verif = data[round(len(data) * 6.6 / 10):]
    labels_train = labels[:round(len(data) * 6.6 / 10)]
    labels_verif = labels[round(len(data) * 6.6 / 10):]
    data_train = data_train.to_numpy()
    data_verif = data_verif.to_numpy()
    labels_train = labels_train.to_numpy()
    labels_verif = labels_verif.to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, 1:].values)
    y = data.iloc[:, 0].values
    #filterring data set 2 train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    freqs = pd.DataFrame({"Training dataset": [(y_train == 1).sum(), (y_train == -1).sum()],
                          "Test dataset": [(y_test == 1).sum(), (y_test == -1).sum()],
                          "Total": [(y_train == 1).sum() + (y_test == 1).sum(),
                                    (y_train == -1).sum() + (y_test == -1).sum()]},
                          index=["Recurrent", "Nonrecurrent"])
    freqs[["Training dataset", "Test dataset", "Total"]]
    #Backpropogation
    '''we use Multi-layer Perceptron .
    MLP is a supervised learning algorithm that learns a function f by training on data set
    The advantages of Multi-layer Perceptron are:

Capability to learn non-linear models.

Capability to learn models in real-time (on-line learning) using partial_fit'''

    #training model
    st = time.time()
    clf = MLPClassifier(solver='adam',hidden_layer_sizes=(45),max_iter=50,activation='relu',random_state=random_state)
    clf.fit(X_train, y_train)
    print("The time for fitting model is: %.2f sec" % (time.time() - st))
    #training results
    y_predict = clf.predict(X_train)
    print("Accuracy of  Backpropagation train is:  %.2f precents" % (metrics.accuracy_score(y_train, y_predict) * 100))
    #testing results
    y_predict = clf.predict(X_test)
    print("Accuracy of  Backpropagation split is:  %.2f precents" % (metrics.accuracy_score(y_test, y_predict) * 100))
    cv = cross_val_score(clf, X, y, cv=3, scoring='accuracy').mean()
    print("Accuracy of  Backpropagation cross-validation is: %.2f precents" % (cv * 100))
    #time
    bpTime = time.time() - st
    print("The time for getting all model results is: %.2f sec" % bpTime)
    #make confusion matrix for this model
  #  cm = confMat(y_test, y_predict, clf.classes_)
    confusion = metrics.confusion_matrix(y_test, y_predict)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('True Positives (TP):', TP)
    print('True Negatives (TN):', TN)
    print('False Positives (FP):', FP)
    print('False Negarives (FN):', FN)


def confMat(y_true, y_pred, labels, figsize=(7,6)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = '0'
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Blues')
    #plt.savefig(filename)
    plt.show()
    return cm
main()



#finding best param - enter to main for check
    #mlp = MLPClassifier()

    # parameter_space = {
    #     'hidden_layer_sizes': [(100,), (2),(10),(50),(15), (30), (45), (60), (80), (2), (10), (45)],
    #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #     #lbfgs= bacpropogation training algo
    #     #sgd updates parameters using the gradient of the loss function with respect to a parameter that needs adaptation
    #     #Adam is similar to SGD in a sense that it is a stochastic optimizer, but it can automatically adjust the amount to update parameters based on adaptive estimates of lower-order moments.
    #     'solver': ['lbfgs', 'sgd', 'adam'],
    #     'max_iter': [10,50, 100, 500, 1000],
    # }
    # #finding best parmeters going through all parameter space we made using gread search
    # #for more info https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    # clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    # clf.fit(X_train, y_train)
    # # Best paramete set
    # print('Best parameters found:\n', clf.best_params_)
