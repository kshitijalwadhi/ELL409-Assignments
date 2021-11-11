import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import seaborn as sns
import time
sns.set_theme()
random.seed(42)
np.random.seed(42)


def relu(x):  # 1
    return np.maximum(0, x)


def relu_der(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return np.array(x)


def sigmoid(x):  # 2
    return 1/(1+np.exp(-1.*x))


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


def tanh(x):  # 3
    return np.tanh(x)


def tanh_der(x):
    return 1-np.square(np.tanh(x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def chunkarr(arr, n):
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def getAccuracy(y, t):
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    return np.sum((y == t))/len(y)


class NN:
    def __init__(self, early=True, epochs=100, layers=2, nodes=[50, 75], alpha=0.0003, bs=16, eps=0.0005, activation=1, reg=1, lam=0):
        np.random.seed(42)
        self.epochs = epochs
        self.layers = layers
        self.nodes = nodes
        self.alpha = alpha
        self.bs = bs
        self.eps = eps
        self.activation = activation
        self.reg = reg
        self.lam = lam
        self.w = [np.random.randn(NUM_FEATURES, self.nodes[0])]
        self.b = [np.random.randn(self.nodes[0])]
        self.z = [0] * (self.layers + 1)
        self.a = [0] * (self.layers)
        self.dw = [0]*(self.layers+1)
        self.db = [0]*(self.layers+1)
        self.early = early

        for i in range(1, self.layers):
            self.w.append(np.random.randn(self.nodes[i-1], self.nodes[i]))
            self.b.append(np.random.randn(self.nodes[i]))

        self.w.append(0.1*np.random.randn(self.nodes[self.layers-1], 10))
        self.b.append(np.random.randn(10))

    def activation_fn(self, x):
        if self.activation == 1:
            return relu(x)
        elif self.activation == 2:
            return sigmoid(x)
        elif self.activation == 3:
            return tanh(x)

    def der_activation_fn(self, x):
        if self.activation == 1:
            return relu_der(x)
        elif self.activation == 2:
            return sigmoid_der(x)
        elif self.activation == 3:
            return tanh_der(x)

    def f_prop(self, X):
        self.z[0] = np.dot(X, self.w[0]) + self.b[0]
        self.a[0] = self.activation_fn(self.z[0])
        for i in range(1, self.layers):
            self.z[i] = np.dot(self.a[i-1], self.w[i]) + self.b[i]
            self.a[i] = self.activation_fn(self.z[i])
        self.z[self.layers] = np.dot(self.a[self.layers-1], self.w[self.layers]) + self.b[self.layers]

    def back_prop(self, X, t, y):
        temp = y-t
        self.dw[self.layers] = (np.dot(self.a[self.layers-1].T, temp) +
                                self.lam*self.w[self.layers])/self.bs
        self.db[self.layers] = (np.sum(temp, axis=0))/self.bs

        for i in range(self.layers-1, -1, -1):
            aux = np.dot(temp, self.w[i+1].T)
            der = self.der_activation_fn(self.a[i])
            temp = aux*der
            if i > 0:
                self.dw[i] = (np.dot(self.a[i-1].T, temp) + self.lam*self.w[i])/self.bs
            else:
                self.dw[0] = (np.dot(X.T, temp) + self.lam*self.w[0])/self.bs
            self.db[i] = np.sum(temp, axis=0)/self.bs

    def train(self, X, t, log=False):
        t_onehot = np.zeros((len(t), 10))
        t_onehot[np.arange(len(t_onehot)), t] = 1
        t = t_onehot
        trainX = np.array(X[:2700])
        traint = np.array(t[:2700])
        testX = np.array(X[2700:])
        testt = np.array(t[2700:])

        Xchunks = chunkarr(trainX, self.bs)
        tchunks = chunkarr(traint, self.bs)

        # calculate accuracy here
        train_acc_list = []
        train_error_list = []
        test_acc_list = []
        test_error_list = []
        train_pred = self.predict(trainX)
        test_pred = self.predict(testX)

        trainacc = getAccuracy(train_pred, traint)
        testacc = getAccuracy(test_pred, testt)
        trainerror = -1*np.sum(traint*np.log(train_pred))/len(trainX)
        testerror = -1*np.sum(testt*np.log(test_pred))/len(testX)

        train_acc_list.append(trainacc)
        test_acc_list.append(testacc)
        train_error_list.append(trainerror)
        test_error_list.append(testerror)

        for epoch in (range(self.epochs)):
            t1 = time.time()
            for chunk in range(len(Xchunks)):
                Xcon = Xchunks[chunk]
                tcon = tchunks[chunk]

                self.f_prop(Xcon)
                y = softmax(self.z[self.layers])

                # call backprop here
                self.back_prop(Xcon, tcon, y)

                for i in range(self.layers+1):
                    self.w[i] -= self.alpha * self.dw[i]
                    self.b[i] -= self.alpha * self.db[i]

            t2 = time.time()

            # calculate accuracies here
            train_pred = self.predict(trainX)
            test_pred = self.predict(testX)

            trainacc = getAccuracy(train_pred, traint)
            testacc = getAccuracy(test_pred, testt)
            trainerror = -1*np.sum(traint*np.log(train_pred))/len(trainX)
            testerror = -1*np.sum(testt*np.log(test_pred))/len(testX)

            train_acc_list.append(trainacc)
            test_acc_list.append(testacc)
            train_error_list.append(trainerror)
            test_error_list.append(testerror)

            if self.early:
                if trainacc > 0.7:
                    if abs(train_acc_list[-1] - train_acc_list[-2]) < self.eps:
                        if abs(train_acc_list[-2] - train_acc_list[-3]) < self.eps:
                            if abs(train_acc_list[-3] - train_acc_list[-4]) < self.eps:
                                break

            if log:
                print(f"epoch: {epoch+1}/{self.epochs}")
                print(f"time: {round(t2-t1,4)}, training_acc: {round(trainacc,4)}, test_acc: {round(testacc,4)}")

        return train_acc_list, test_acc_list, train_error_list, test_error_list

    def predict(self, X):
        self.z = [0] * (self.layers + 1)
        self.a = [0] * (self.layers)
        self.f_prop(X)
        return softmax(self.z[self.layers])


def plot_acc(train_acc_list, test_acc_list, name):
    xa = np.linspace(1, len(train_acc_list), len(train_acc_list), dtype=int)
    plt.figure()
    train_acc_list = np.array(train_acc_list) * 100
    test_acc_list = np.array(test_acc_list) * 100
    plt.plot(xa, train_acc_list, color="g", label="Training accuracy")
    plt.plot(xa, test_acc_list, color="r", label="Testing accuracy")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.legend()
    plt.savefig(f'Accuracy_{name}.png', dpi=1200)
    plt.show()


def plot_error(train_err_list, test_err_list, name):
    xa = np.linspace(1, len(train_err_list), len(train_err_list), dtype=int)
    plt.figure()
    plt.plot(xa, train_err_list, color="g", label="Training error")
    plt.plot(xa, test_err_list, color="r", label="Testing error")
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error vs Iterations')
    plt.legend()
    plt.savefig(f'Error_{name}.png', dpi=1200)
    plt.show()
