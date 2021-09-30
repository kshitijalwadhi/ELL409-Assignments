import random
import math
import numpy as np
import argparse
import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm
np.random.seed(42)


def readData(filename):
    dataset = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dataset.append([float(row[0]), float(row[1])])
    return dataset


# def readDataLink(filename):
#     dataset = []
#     df = pd.read_csv(filename, header=None)
#     for i, row in df.iterrows():
#         dataset.append([float(row[0]), float(row[1])])
#     return dataset


# highest power is m here
def makeDesignMat(x, m):
    return np.vander(x, m+1, increasing=True)


meanArr = []
sigmaArr = []


# # highest power is m here
# def makeDesignMat(x, m):
#     global meanArr
#     global sigmaArr
#     temp = np.vander(x, m+1, increasing=True)
#     meanArr = temp.mean(0)
#     sigmaArr = np.std(temp-meanArr, axis=0)
#     meanArr[0] = 0.0
#     sigmaArr[0] = 1.0
#     temp = (temp-meanArr)/sigmaArr
#     return temp


def makeDesignMatForTest(x, m):
    global meanArr
    global sigmaArr
    temp = np.vander(x, m+1, increasing=True)
    temp = (temp-meanArr)/sigmaArr
    return temp


def denormweights(w):
    origw = w/sigmaArr
    temp = 0
    for i in range(len(w)):
        temp += w[i] * meanArr[i]/sigmaArr[i]
    origw[0] -= temp
    return origw


def chunkarr(arr, n):
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def descent(trainX, y, testX, testY, lambd=0.0, err_function="mse"):
    w = np.zeros(MAX_POWER+1)
    bestw = np.zeros(MAX_POWER+1)
    y = np.array(y)
    X = makeDesignMat(trainX, MAX_POWER)
    Xchunks = chunkarr(X, BATCH_SIZE)
    ychunks = chunkarr(y, BATCH_SIZE)
    train_err_list = []
    test_err_list = []
    lr = LEARNING_RATE
    for j in (range(NUM_ITER)):
        #preverr = testError(trainX, y, w, MAX_POWER)
        for chunk in range(len(Xchunks)):
            Xcon = Xchunks[chunk]
            ycon = ychunks[chunk]
            pred = np.dot(Xcon, w)
            test_err_list.append(testError(testX, testY, w, MAX_POWER))
            if err_function == "mse":
                err = 0.5 * (np.sum(np.square(pred-ycon)) + lambd*np.sum(np.square(w)))/len(ycon)
                #grad = (np.dot(Xcon.T, (pred-ycon)) + lambd*w)/len(ycon)
                grad = (np.dot(Xcon.T, (pred-ycon)) + lambd*w)
                grad /= np.linalg.norm(grad)
            if err_function == "mae":
                err = np.sum(np.abs(pred-ycon))
                parity = np.sign(pred - ycon)
                parity = parity.reshape((len(ycon), 1))
                grad = np.sum(np.multiply(parity, Xcon).T, axis=1)/len(ycon)
            if err_function == "huber":
                e = 0.1
                temp = pred-ycon
                err = np.sum(np.where(np.abs(temp) <= e, 0.5*temp*temp, (e*np.abs(temp) - 0.5*e*e)))
                err += lambd*np.sum(np.square(w))
                err /= len(ycon)
                grad = np.zeros((MAX_POWER+1,))
                for i in range(len(ycon)):
                    if np.abs(temp[i] <= e):
                        grad += temp[i]*Xcon[i, :]
                    else:
                        grad += e*np.sign(temp[i])*Xcon[i, :]
                grad += 2*lambd*w
                w = np.squeeze(w)
                grad /= len(ycon)
                grad = np.squeeze(grad)
            w = w - lr * grad
            train_err_list.append(err)
        #newerr = testError(trainX, y, w, MAX_POWER)
        # if newerr > preverr:
        #     return w, False
        # if(j % 5000 == 0):
        #     print(newerr)
        #lr = LEARNING_RATE * math.exp(-0.0015*j/NUM_ITER)
    return w, True


def testError(X, Y, w, MAX_POWER):
    X = makeDesignMat(X, MAX_POWER)
    pred = np.dot(X, w)
    if ERROR == "mse":
        err = 0.5 * (np.sum(np.square(pred-Y)))/len(Y)
    elif ERROR == "mae":
        err = np.sum(np.abs(pred-Y)) / len(Y)
    elif ERROR == "huber":
        e = 0.1
        temp = pred-Y
        err = np.sum(np.where(np.abs(temp) <= e, 0.5*temp*temp, (e*np.abs(temp) - 0.5*e*e))) / len(Y)
    return err


def evaluateGD(dataset):
    dataset = np.array(dataset)
    numtr = int(SUBSET * SPLIT)
    trainX = dataset[:, 0][:numtr]
    trainY = dataset[:, 1][:numtr]
    testX = dataset[:, 0][numtr:]
    testY = dataset[:, 1][numtr:]
    w, flag = descent(trainX, trainY, testX, testY, REGULARIZATION_LAMBDA, ERROR)
    # if flag:
    #     plot_reg(trainX, trainY, testX, testY, w)
    trainerr = testError(trainX, trainY, w, MAX_POWER)
    testerr = testError(testX, testY, w, MAX_POWER)
    return w, trainerr, testerr, flag


def evaluatePINV(dataset):
    dataset = np.array(dataset)
    numtr = int(SUBSET * SPLIT)
    trainX = dataset[:, 0][:numtr]
    trainY = dataset[:, 1][:numtr]
    testX = dataset[:, 0][numtr:]
    testY = dataset[:, 1][numtr:]
    X = makeDesignMat(trainX, MAX_POWER)
    w = np.dot(np.linalg.inv(np.dot(X.transpose(), X) + REGULARIZATION_LAMBDA * np.identity(MAX_POWER+1)), np.dot(X.transpose(), trainY))
    #plot_reg(trainX, trainY, testX, testY, w)
    trainerr = testError(trainX, trainY, w, MAX_POWER)
    testerr = testError(testX, testY, w, MAX_POWER)
    return w, trainerr, testerr


# def plot_reg(trainX, trainY, testX, testY, w):
#     def aux(X_line, w):
#         temp = 0
#         for i in range(len(w)):
#             temp += np.power(X_line, i) * w[i]
#         return temp
#     jointX = np.concatenate((trainX, testX))
#     plt.clf()
#     plt.scatter(trainX, trainY, color='b', marker='o', s=15)
#     plt.scatter(testX, testY, color="m", marker='o', s=30)
#     x_line = np.linspace(min(jointX), max(jointX), 100)
#     y_pred = aux(x_line, w)
#     plt.plot(x_line, y_pred, color='g')
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.show()
#     return plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default=1, help="which part")
    parser.add_argument("--method", default="pinv", help="type of solver")
    parser.add_argument("--batch_size", default=5, type=int, help="batch size")
    parser.add_argument("--lamb", default=0, type=float, help="regularization constant")
    parser.add_argument("--X", default="", type=str, help="Read content from the file")
    parser.add_argument("--polynomial", default=10, type=float, help="degree of polynomial")

    args = parser.parse_args()

    dataset = readData(args.X)
    # dataset = readDataLink('https://web.iitd.ac.in/~sumeet/A1/2019EE10577/gaussian.csv')
    MAX_POWER = int(args.polynomial)
    BATCH_SIZE = args.batch_size
    REGULARIZATION_LAMBDA = args.lamb
    ERROR = "mse"

    LEARNING_RATE = 0.003
    #LEARNING_RATE = 1

    NUM_ITER = 1000000
    if BATCH_SIZE == 1:
        NUM_ITER = 70000
    elif BATCH_SIZE == 2:
        NUM_ITER = 150000
    elif BATCH_SIZE <= 5:
        NUM_ITER = 250000
    elif BATCH_SIZE <= 14:
        NUM_ITER /= 2
    NUM_ITER = int(math.floor(NUM_ITER))
    #NUM_ITER = 100000

    SPLIT = 0.9
    SUBSET = len(dataset)

    if args.method == "gd":
        w, trainerr, testerr, flag = evaluateGD(dataset)
        # while(True):
        #     print(f"learning rate is:{LEARNING_RATE}")
        #     w, trainerr, testerr, flag = evaluateGD(dataset)
        #     if flag == True:
        #         break
        #     LEARNING_RATE = 0.95*LEARNING_RATE
    elif args.method == "pinv":
        w, trainerr, testerr = evaluatePINV(dataset)

    print(f"weights={w}")
