from datetime import time
import matplotlib.pyplot as plt
import time
import warnings
import matplotlib
from numpy.random import seed
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
random_state = 42
warnings.filterwarnings('ignore')

class Adaline(object):
    """Adaptive Linear Neuron Classifier.
    Parameters:
    eta learning rate(between 0 and 1)
    n_iter : Passes over the training dataset.
    """
    def __init__(self, eta=0.005, n_iter=5, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        if random_state:
            # Set random state for shuffling and initializing the weights
            seed(random_state)

    def get_params(self, deep=True):
        return {"n_iter": self.n_iter, "eta": self.eta}

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        X : shape = [n_samples, n_features]. The Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : shape = [n_samples]. Target values('lables').
        """
        #make array of weights size n_sampls+1
        self.initWeights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            cost = []
            #
            for xi, target in zip(X, y):
                cost.append(self.updateWeights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    """Fit training data without reinitializing the weights"""
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self.initWeights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.updateWeights(xi, target)
        else:
            self.updateWeights(X, y)
        return self

    def initWeights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def updateWeights(self, xi, target):
        #Apply Adaline learning rule to update the weights
        #mul Wi Xi +t
        output = self.net_input(xi)
        #maybe target -error**2
        error = target - output
        ## Update the weights based on the delta rule
         #Wi= Wi +lrate*error*Xi
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost
    #Calculate net input
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    #Compute linear activation
    def activation(self, X):
        return self.net_input(X)
    #Return the label after fit
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

def main():
    data = pd.read_csv('wpbc.csv')
    data = data.replace(["N", "R"], [-1, 1])
    labels = data[["b"]]
    #dropped after checking features in cor mat
    data = data.drop(["a","c","f","g","i","p","s","x","z"], axis=1)
    data = data.replace(['?'], [6])
    #fit for train test
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, 1:].values)
    y = data.iloc[:, 0].values
    #split to train test using sklearn.model_selection._split` module includes functions to split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
    #show the dataset split
    freqs = pd.DataFrame({"Training dataset": [(y_train == 1).sum(), (y_train == -1).sum()],
                          "Test dataset": [(y_test == 1).sum(), (y_test == -1).sum()],
                          "Total": [(y_train == 1).sum() + (y_test == 1).sum(),
                                    (y_train == -1).sum() + (y_test == -1).sum()]},
                          index=["Recurrent", "Nonrecurrent"])
    freqs[["Training dataset", "Test dataset", "Total"]]
    print(freqs)
    #training the model
    st = time.time()
    ada = Adaline(n_iter=7, eta=0.005, random_state=random_state)
    ada.fit(X_train, y_train)
    tfit= time.time() - st
    print("The time for model fitting is: %.2f sec" % tfit)

    #training results
    y_predict = ada.predict(X_train)
    print("Accuracy of Adaline (train): %.2f percents" % (metrics.accuracy_score(y_train, y_predict) * 100))

    #testing results
    y_predict = ada.predict(X_test)
    print("Accuracy of Adaline (split): %.2f percents" % (metrics.accuracy_score(y_test, y_predict) * 100))
   #cross validation using sklearn cross val func
    cv = cross_val_score(ada, X, y, cv=3, scoring='accuracy').mean()
    print("Accuracy of Adaline (cross-validation): %.2f percents" % (cv * 100))
    print("Standart Deviation of Adaline (cross-validation) %.2f precents" % (cross_val_score(ada, X, y, cv=3, scoring='accuracy').std() * 100))
    adaTime = time.time() - st
    print("The time for getting all the model results: %.2f sec" % (adaTime))


main()


#con mat fubction
# def cm_analysis(y_true, y_pred, labels, figsize=(7,6)):
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     cm_sum = np.sum(cm, axis=1, keepdims=True)
#     cm_perc = cm / cm_sum.astype(float) * 100
#     annot = np.empty_like(cm).astype(str)
#     nrows, ncols = cm.shape
#     for i in range(nrows):
#         for j in range(ncols):
#             c = cm[i, j]
#             p = cm_perc[i, j]
#             if i == j:
#                 s = cm_sum[i]
#                 annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
#             elif c == 0:
#                 annot[i, j] = '0'
#             else:
#                 annot[i, j] = '%.1f%%\n%d' % (p, c)
#     cm = pd.DataFrame(cm, index=labels, columns=labels)
#     cm.index.name = 'Actual'
#     cm.columns.name = 'Predicted'
#     fig, ax = plt.subplots(figsize=figsize)
#     sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Blues')
#     #plt.savefig(filename)
#     plt.show()
#     return cm
# def confMat(y_true, y_pred, labels, figsize=(7,7)):
#    #make conusion matrix usingg skl function
#     #for more info https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     cm_sum = np.sum(cm, axis=1, keepdims=True)
#     cm_perc = cm / cm_sum.astype(float) * 100
#     annot = np.empty_like(cm).astype(str)
#     nrows, ncols = cm.shape
#     for i in range(nrows):
#         for j in range(ncols):
#             c = cm[i, j]
#             p = cm_perc[i, j]
#             if i == j:
#                 s = cm_sum[i]
#                 annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
#             elif c == 0:
#                 annot[i, j] = '0'
#             else:
#                 annot[i, j] = '%.1f%%\n%d' % (p, c)
#     cm = pd.DataFrame(cm, index=labels, columns=labels)
#     cm.index.name = 'Actual'
#     cm.columns.name = 'Predicted'
#     fig, ax = plt.subplots(figsize=figsize)
#     sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='Blues')
#     #plt.savefig(filename)
#     plt.show()
#     return cm
# #make confusion matrix for this run
# cm = confMat(y_test, y_predict, [-1,1])
# """Finding best parameters for Adaline"""- add to main for run
    # i_range = range(1, 41)
    # l_range = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.1, 0.025, 0.6, 0.003]
    # zdata = []
    # xdata = []
    # ydata = []
    # for i in i_range:
    #     #check every l_rangechecking it in adaline and add cv score , iter_n , eta  to list
    #     for l in l_range:
    #         ada = Adaline(n_iter=i, eta=l, random_state=random_state)
    #         ada.fit(X_train, y_train)
    #         y_predict = ada.predict(X_test)
    #         cv = cross_val_score(ada, X, y, cv=3, scoring='accuracy').mean()
    #         zdata.append(cv)
    #         xdata.append(i)
    #         ydata.append(l)
    # matplotlib
    # ##show it using 3D view
    # fig = plt.figure(figsize=(10, 10))
    # #https://matplotlib.org / mpl_toolkits / mplot3d / tutorial.html
    # ax = plt.axes(projection='3d')
    # #we choose the color cmap
    # #for more options see: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    # ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis');
    # fig.show()
    # #print best score
    # x = 0
    # for i in zdata:
    #     if (max(zdata) == i):
    #         break
    #     x += 1
    # print("best score: ", zdata[x])
    # print("n_iter: ", xdata[x])
    # print("eta: ", ydata[x])
