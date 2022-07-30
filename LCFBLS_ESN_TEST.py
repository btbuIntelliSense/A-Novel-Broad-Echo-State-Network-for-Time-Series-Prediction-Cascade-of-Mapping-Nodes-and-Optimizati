import pandas as pd
import numpy as np
from BLS import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from CFBLS import CFBLS
from LCFBLS import LCFBLS
from CEBLS import CEBLS
from LCFBLS_ESN import LCFBLS_ESN
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False


def getDataFromCSV(path):
    '''
    将数据切片
    :return: 没有归一化的数据
    '''
    data = pd.read_csv(path).values[1000:16000, :].reshape(15000, 7)

    initLen = 0

    label = data[:, :1].reshape(-1, 1)
    data = data[:, 1: 7]
    # print(data.shape, max(label) + 1)
    # traindata , testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.01,random_state = 0)

    trainLen = 12000
    testLen = len(data) - trainLen
    traindata = data[initLen:trainLen, :]
    trainlabel = label[initLen:trainLen]
    testdata = data[trainLen: trainLen + testLen, :]
    testlabel = label[trainLen: trainLen + testLen]
    return traindata, trainlabel, testdata, testlabel

def getMAE(predict, real):
    return mean_absolute_error(real, predict)


def getSMAPE(predict, real):
    return np.mean(np.abs(predict - real) / (np.abs(predict) + np.abs(real)))


def getMSE(predict, real):
    return mean_squared_error(real, predict)


def getRMSE(predict, real):
    MSE = getMSE(predict, real)
    return np.sqrt(MSE)


def getR2(predict, real):
    # average = np.sum(real) / len(real)
    # return 1 - (np.sum(np.dot((real - predict).T, (real - predict))) / np.sum(np.dot((real - average).T, (real - average))))
    return r2_score(real, predict)


def getR(predict, real):
    '''
    以下是绝对值的区间范围，R本身取值在[-1，1]之间
    0.8-1.0     极强相关
    0.6-0.8     强相关
    0.4-0.6     中等程度相关
    0.2-0.4     弱相关
    0.0-0.2     极弱相关或无相关
    '''
    predict = np.squeeze(predict)  # 去掉多余的维度
    real = np.squeeze(real)
    return pearsonr(real, predict)[0]

if __name__ == '__main__':
    path = 'E:\\yan_1\\BLS_self\\fangshan.csv'
    traindata, trainlabel, testdata, testlabel = getDataFromCSV(path)
    initLength = 100

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scaler3 = MinMaxScaler()
    scaler4 = MinMaxScaler()

    traindata = scaler1.fit_transform(traindata)
    trainlabel = scaler2.fit_transform(trainlabel)
    testdata = scaler3.fit_transform(testdata)
    testlabel = scaler4.fit_transform(testlabel)

    '''
    s------收敛系数
    c------正则化系数
    N1-----映射层每个窗口内节点数
    N2-----映射层窗口数
    N3-----强化层节点数
    L------增加强化层强化窗口数
    M------每个强化层窗口的强化节点个数
    '''
    #实验BLS案例
    BLS_predictTrain, BLS_predictTest = BLS(traindata, trainlabel, testdata, testlabel,s=0.8, c=2**-28, N1=20, N2=20, N3=10)
    CFBLS_predictTrain, CFBLS_predictTest = CFBLS(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20, N2=20,N3=10)
    LCFBLS_predictTrain, LCFBLS_predictTest = LCFBLS(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20,N2=20,N3=10)
    CEBLS_predictTrain, CEBLS_predictTest = CEBLS(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20,N2=20,N3=10)
    LCFBLS_ESN_predictTrain, LCFBLS_ESN_predictTest = LCFBLS_ESN(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20,N2=20,N3=10)

    BLS_predictTrain = scaler2.inverse_transform(BLS_predictTrain)
    BLS_predictTest = scaler4.inverse_transform(BLS_predictTest)
    CFBLS_predictTrain = scaler2.inverse_transform(CFBLS_predictTrain)
    CFBLS_predictTest = scaler4.inverse_transform(CFBLS_predictTest)
    LCFBLS_predictTrain = scaler2.inverse_transform(LCFBLS_predictTrain)
    LCFBLS_predictTest = scaler4.inverse_transform(LCFBLS_predictTest)
    CEBLS_predictTrain = scaler2.inverse_transform(CEBLS_predictTrain)
    CEBLS_predictTest = scaler4.inverse_transform(CEBLS_predictTest)
    LCFBLS_ESN_predictTrain = scaler2.inverse_transform(LCFBLS_ESN_predictTrain)
    LCFBLS_ESN_predictTest = scaler4.inverse_transform(LCFBLS_ESN_predictTest)
    trainlabel = scaler2.inverse_transform(trainlabel)
    testlabel = scaler4.inverse_transform(testlabel)

    BLS_TrainRMSE = getRMSE(BLS_predictTrain[initLength:], trainlabel[initLength:])
    BLS_TestRMSE = getRMSE(BLS_predictTest[initLength:], testlabel[initLength:])
    CFBLS_TrainRMSE = getRMSE(CFBLS_predictTrain[initLength:], trainlabel[initLength:])
    CFBLS_TestRMSE = getRMSE(CFBLS_predictTest[initLength:], testlabel[initLength:])
    LCFBLS_TrainRMSE = getRMSE(LCFBLS_predictTrain[initLength:], trainlabel[initLength:])
    LCFBLS_TestRMSE = getRMSE(LCFBLS_predictTest[initLength:], testlabel[initLength:])
    CEBLS_TrainRMSE = getRMSE(CEBLS_predictTrain[initLength:], trainlabel[initLength:])
    CEBLS_TestRMSE = getRMSE(CEBLS_predictTest[initLength:], testlabel[initLength:])
    LCFBLS_ESN_TrainRMSE = getRMSE(LCFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
    LCFBLS_ESN_TestRMSE = getRMSE(LCFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

    print("BLS_TrainRMSE : ", BLS_TrainRMSE)
    print("BLS_TestRMSE", BLS_TestRMSE)
    print("CFBLS_TrainRMSE : ", CFBLS_TrainRMSE)
    print("CFBLS_TestRMSE", CFBLS_TestRMSE)
    print("LCFBLS_TrainRMSE : ", LCFBLS_TrainRMSE)
    print("LCFBLS_TestRMSE", LCFBLS_TestRMSE)
    print("CEBLS_TrainRMSE : ", CEBLS_TrainRMSE)
    print("CEBLS_TestRMSE", CEBLS_TestRMSE)
    print("LCFBLS_ESN_TrainRMSE : ", LCFBLS_ESN_TrainRMSE)
    print("LCFBLS_ESN_TestRMSE", LCFBLS_ESN_TestRMSE)

    fig = plt.figure(figsize=(13, 5))
    # x = np.arange(len(testlabel)-initLength)
    x = np.arange(200)
    plt.plot(x, testlabel[initLength:initLength+200], color='#FF0000', label='real')
    # plt.plot(x, BLS_predictTest[:600], color='#00B0F0', label='BLS')
    plt.plot(x, CFBLS_predictTest[initLength:initLength+200], color='#CD853F', label='CFBLS')
    plt.plot(x, LCFBLS_predictTest[initLength:initLength+200], color='#00B050', label='LCFBLS')
    # plt.plot(x, CEBLS_predictTest[:600], color='#92D050', label='CEBLS')
    plt.plot(x, LCFBLS_ESN_predictTest[initLength:initLength+200], color='#92D050', label='LCFBLS_ESN')
    plt.legend(loc='upper left')  # 把图例设置在外边
    plt.ylabel('AQI')
    plt.show()
