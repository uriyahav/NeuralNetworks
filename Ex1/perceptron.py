from __future__ import print_function
import time
import matplotlib, sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
##try to predict Wi*Xi
def predict(inputs, weights):

    activation = 0.0
    for i, w in zip(inputs, weights):
        activation += i*w
    return 1.0 if activation >= 0.0 else 0.0

def accuracy(matrix, weights):
    num_correct = 0.0
    preds = []
    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1], weights)  # get predicted classification
        preds.append(pred)
        if pred == matrix[i][-1]: num_correct += 1.0
    #print("Predictions:", preds)
    return num_correct / float(len(matrix))


def train_weights(matrix, weights,times, l_rate=1.00, do_plot=True, stop_early=True, verbose=True):

    for epoch in range(times):
        cur_acc = accuracy(matrix, weights)
        #if do_plot: plot(matrix, weights, title="Epoch %d" % epoch)
        print("\nRun %d \nWeights: " % epoch, weights)
        print("Accuracy: ", cur_acc)

        if cur_acc > 0.75 and stop_early: break


        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1], weights)  # get predicted classificaion
            error = matrix[i][-1] - prediction  # get error from real classification
            for j in range(len(weights)):  # calculate new weight for each node
               weights[j] = weights[j] + (l_rate * error * matrix[i][j])

    return weights


def main():
    times = 10
    l_rate = 0.01
    do_plot = True
    stop_early = True
    data = pd.read_csv('wpbc.csv')
    #formatting the data
    data['result'] = np.where(data.iloc[:, 1] == 'N', 0, 1)
    data = data.replace(["N", "R"], [-1, 1])
    labels = data[["b"]]
    #,"c", "f", "g", "i", "p", "q",
    data = data.drop(["a","s", "x", "z", "aa"], axis=1)
    data = data.replace(['?'], [6])
    #split the data for train and verify
    data_train = data[:round(len(data) * 6.6 / 10)]
    data_verif = data[round(len(data) * 6.6 / 10):]
    labels_train = data_train[["b"]]
    data_train = data_train.to_numpy().astype(float)
    data_verif = data_verif.to_numpy().astype(float)

    s= time.time()
    weights = np.zeros(30)
    weights = train_weights(data_train ,weights ,times ,l_rate ,do_plot ,stop_early)
    # plt.plot(labels)
    # plt.show()

    adaTime = time.time() - s
    print("The time for training the model is: %.2f sec" % (adaTime))
    #test
    s=time.time()
    print("accuracy of test:")
    print(accuracy(data_verif, weights))
    adaTime2 = time.time() - s
    print("The time for Testing the model is: %.2f sec" % (adaTime2))
    print("Total time for this model is: %.2f sec" % (adaTime2+adaTime))


main()
