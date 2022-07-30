import pandas as pd
import numpy as np
from BLS import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes
from CFBLS import CFBLS
from LCFBLS import LCFBLS
from CEBLS import CEBLS
from LCFBLS_ESN import LCFBLS_ESN
from CFBLS_ESN import CFBLS_ESN
from BLS_ESN import BLS_ESN
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

    mappingNumberList = []
    enhanceNumberList = []


    BLS_RMSE_TRAIN = []
    BLS_RMSE_TEST = []
    CFBLS_RMSE_TRAIN = []
    CFBLS_RMSE_TEST = []
    LCFBLS_RMSE_TRAIN = []
    LCFBLS_RMSE_TEST = []
    CEBLS_RMSE_TRAIN = []
    CEBLS_RMSE_TEST = []
    BLS_ESN_RMSE_TRAIN = []
    BLS_ESN_RMSE_TEST = []
    CFBLS_ESN_RMSE_TRAIN = []
    CFBLS_ESN_RMSE_TEST = []
    LCFBLS_ESN_RMSE_TRAIN = []
    LCFBLS_ESN_RMSE_TEST = []

    BLS_MAE_TRAIN = []
    BLS_MAE_TEST = []
    CFBLS_MAE_TRAIN = []
    CFBLS_MAE_TEST = []
    LCFBLS_MAE_TRAIN = []
    LCFBLS_MAE_TEST = []
    CEBLS_MAE_TRAIN = []
    CEBLS_MAE_TEST = []
    BLS_ESN_MAE_TRAIN = []
    BLS_ESN_MAE_TEST = []
    CFBLS_ESN_MAE_TRAIN = []
    CFBLS_ESN_MAE_TEST = []
    LCFBLS_ESN_MAE_TRAIN = []
    LCFBLS_ESN_MAE_TEST = []

    BLS_SMAPE_TRAIN = []
    BLS_SMAPE_TEST = []
    CFBLS_SMAPE_TRAIN = []
    CFBLS_SMAPE_TEST = []
    LCFBLS_SMAPE_TRAIN = []
    LCFBLS_SMAPE_TEST = []
    CEBLS_SMAPE_TRAIN = []
    CEBLS_SMAPE_TEST = []
    BLS_ESN_SMAPE_TRAIN = []
    BLS_ESN_SMAPE_TEST = []
    CFBLS_ESN_SMAPE_TRAIN = []
    CFBLS_ESN_SMAPE_TEST = []
    LCFBLS_ESN_SMAPE_TRAIN = []
    LCFBLS_ESN_SMAPE_TEST = []

    BLS_R2_TRAIN = []
    BLS_R2_TEST = []
    CFBLS_R2_TRAIN = []
    CFBLS_R2_TEST = []
    LCFBLS_R2_TRAIN = []
    LCFBLS_R2_TEST = []
    CEBLS_R2_TRAIN = []
    CEBLS_R2_TEST = []
    BLS_ESN_R2_TRAIN = []
    BLS_ESN_R2_TEST = []
    CFBLS_ESN_R2_TRAIN = []
    CFBLS_ESN_R2_TEST = []
    LCFBLS_ESN_R2_TRAIN = []
    LCFBLS_ESN_R2_TEST = []

    BLS_R_TRAIN = []
    BLS_R_TEST = []
    CFBLS_R_TRAIN = []
    CFBLS_R_TEST = []
    LCFBLS_R_TRAIN = []
    LCFBLS_R_TEST = []
    CEBLS_R_TRAIN = []
    CEBLS_R_TEST = []
    BLS_ESN_R_TRAIN = []
    BLS_ESN_R_TEST = []
    CFBLS_ESN_R_TRAIN = []
    CFBLS_ESN_R_TEST = []
    LCFBLS_ESN_R_TRAIN = []
    LCFBLS_ESN_R_TEST = []

    BLS_TIME_TRAIN = []
    BLS_TIME_TEST = []
    CFBLS_TIME_TRAIN = []
    CFBLS_TIME_TEST = []
    LCFBLS_TIME_TRAIN = []
    LCFBLS_TIME_TEST = []
    CEBLS_TIME_TRAIN = []
    CEBLS_TIME_TEST = []
    BLS_ESN_TIME_TRAIN = []
    BLS_ESN_TIME_TEST = []
    CFBLS_ESN_TIME_TRAIN = []
    CFBLS_ESN_TIME_TEST = []
    LCFBLS_ESN_TIME_TRAIN = []
    LCFBLS_ESN_TIME_TEST = []

    for i in range(20, 40, 2):
        for j in range(10 ,40, 2):
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

            BLS_predictTrain, BLS_predictTest, BLS_TRAIN_TIME, BLS_TEST_TIME = BLS(traindata, trainlabel, testdata, testlabel,s=0.8, c=2**-28, N1=20, N2=i, N3=j)
            CFBLS_predictTrain, CFBLS_predictTest, CFBLS_TRAIN_TIME, CFBLS_TEST_TIME = CFBLS(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20, N2=i,N3=j)
            LCFBLS_predictTrain, LCFBLS_predictTest, LCFBLS_TRAIN_TIME, LCFBLS_TEST_TIME = LCFBLS(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20,N2=i,N3=j)
            CEBLS_predictTrain, CEBLS_predictTest, CEBLS_TRAIN_TIME, CEBLS_TEST_TIME = CEBLS(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20,N2=i,N3=j)
            LCFBLS_ESN_predictTrain, LCFBLS_ESN_predictTest, LCFBLS_ESN_TRAIN_TIME,LCFBLS_ESN_TEST_TIME = LCFBLS_ESN(traindata, trainlabel, testdata, testlabel, s=0.8, c=2 ** -28, N1=20,N2=i,N3=j)
            BLS_ESN_predictTrain, BLS_ESN_predictTest, BLS_ESN_TRAIN_TIME, BLS_ESN_TEST_TIME = BLS_ESN(traindata, trainlabel, testdata, testlabel, s=0.8,
                                                                         c=2 ** -28, N1=20, N2=i, N3=j)
            CFBLS_ESN_predictTrain, CFBLS_ESN_predictTest, CFBLS_ESN_TRAIN_TIME, CFBLS_ESN_TEST_TIME = CFBLS_ESN(traindata, trainlabel, testdata, testlabel, s=0.8,
                                                                         c=2 ** -28, N1=20, N2=i, N3=j)

            BLS_TIME_TRAIN.append(BLS_TRAIN_TIME)
            BLS_TIME_TEST.append(BLS_TEST_TIME)
            CFBLS_TIME_TRAIN.append(CFBLS_TRAIN_TIME)
            CFBLS_TIME_TEST.append(CFBLS_TEST_TIME)
            LCFBLS_TIME_TRAIN.append(LCFBLS_TRAIN_TIME)
            LCFBLS_TIME_TEST.append(LCFBLS_TEST_TIME)
            CEBLS_TIME_TRAIN.append(CEBLS_TRAIN_TIME)
            CEBLS_TIME_TEST.append(CEBLS_TEST_TIME)
            BLS_ESN_TIME_TRAIN.append(BLS_ESN_TRAIN_TIME)
            BLS_ESN_TIME_TEST.append(BLS_ESN_TEST_TIME)
            CFBLS_ESN_TIME_TRAIN.append(CFBLS_ESN_TRAIN_TIME)
            CFBLS_ESN_TIME_TEST.append(CFBLS_ESN_TEST_TIME)
            LCFBLS_ESN_TIME_TRAIN.append(LCFBLS_ESN_TRAIN_TIME)
            LCFBLS_ESN_TIME_TEST.append(LCFBLS_ESN_TEST_TIME)

            BLS_predictTrain = scaler2.inverse_transform(BLS_predictTrain)
            BLS_predictTest = scaler4.inverse_transform(BLS_predictTest)

            CFBLS_predictTrain = scaler2.inverse_transform(CFBLS_predictTrain)
            CFBLS_predictTest = scaler4.inverse_transform(CFBLS_predictTest)

            LCFBLS_predictTrain = scaler2.inverse_transform(LCFBLS_predictTrain)
            LCFBLS_predictTest = scaler4.inverse_transform(LCFBLS_predictTest)

            CEBLS_predictTrain = scaler2.inverse_transform(CEBLS_predictTrain)
            CEBLS_predictTest = scaler4.inverse_transform(CEBLS_predictTest)

            CFBLS_ESN_predictTrain = scaler2.inverse_transform(CFBLS_ESN_predictTrain)
            CFBLS_ESN_predictTest = scaler4.inverse_transform(CFBLS_ESN_predictTest)

            BLS_ESN_predictTrain = scaler2.inverse_transform(BLS_ESN_predictTrain)
            BLS_ESN_predictTest = scaler4.inverse_transform(BLS_ESN_predictTest)

            LCFBLS_ESN_predictTrain = scaler2.inverse_transform(LCFBLS_ESN_predictTrain)
            LCFBLS_ESN_predictTest = scaler4.inverse_transform(LCFBLS_ESN_predictTest)

            traindata = scaler1.inverse_transform(traindata)
            testdata = scaler3.inverse_transform(testdata)
            trainlabel = scaler2.inverse_transform(trainlabel)
            testlabel = scaler4.inverse_transform(testlabel)

            #计算RMSE
            BLS_TrainRMSE = getRMSE(BLS_predictTrain[initLength:], trainlabel[initLength:])
            BLS_TestRMSE = getRMSE(BLS_predictTest[initLength:], testlabel[initLength:])

            CFBLS_TrainRMSE = getRMSE(CFBLS_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_TestRMSE = getRMSE(CFBLS_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_TrainRMSE = getRMSE(LCFBLS_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_TestRMSE = getRMSE(LCFBLS_predictTest[initLength:], testlabel[initLength:])

            CEBLS_TrainRMSE = getRMSE(CEBLS_predictTrain[initLength:], trainlabel[initLength:])
            CEBLS_TestRMSE = getRMSE(CEBLS_predictTest[initLength:], testlabel[initLength:])

            BLS_ESN_TrainRMSE = getRMSE(BLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            BLS_ESN_TestRMSE = getRMSE(BLS_ESN_predictTest[initLength:], testlabel[initLength:])

            CFBLS_ESN_TrainRMSE = getRMSE(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_ESN_TestRMSE = getRMSE(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_ESN_TrainRMSE = getRMSE(LCFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_ESN_TestRMSE = getRMSE(LCFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            # 计算MAE
            BLS_TrainMAE = getMAE(BLS_predictTrain[initLength:], trainlabel[initLength:])
            BLS_TestMAE = getMAE(BLS_predictTest[initLength:], testlabel[initLength:])

            CFBLS_TrainMAE = getMAE(CFBLS_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_TestMAE = getMAE(CFBLS_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_TrainMAE = getMAE(LCFBLS_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_TestMAE = getMAE(LCFBLS_predictTest[initLength:], testlabel[initLength:])

            CEBLS_TrainMAE = getMAE(CEBLS_predictTrain[initLength:], trainlabel[initLength:])
            CEBLS_TestMAE = getMAE(CEBLS_predictTest[initLength:], testlabel[initLength:])

            BLS_ESN_TrainMAE = getMAE(BLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            BLS_ESN_TestMAE = getMAE(BLS_ESN_predictTest[initLength:], testlabel[initLength:])

            CFBLS_ESN_TrainMAE = getMAE(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_ESN_TestMAE = getMAE(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_ESN_TrainMAE = getMAE(LCFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_ESN_TestMAE = getMAE(LCFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            # 计算SMAPE
            BLS_TrainSMAPE = getSMAPE(BLS_predictTrain[initLength:], trainlabel[initLength:])
            BLS_TestSMAPE = getSMAPE(BLS_predictTest[initLength:], testlabel[initLength:])

            CFBLS_TrainSMAPE = getSMAPE(CFBLS_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_TestSMAPE = getSMAPE(CFBLS_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_TrainSMAPE = getSMAPE(LCFBLS_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_TestSMAPE = getSMAPE(LCFBLS_predictTest[initLength:], testlabel[initLength:])

            CEBLS_TrainSMAPE = getSMAPE(CEBLS_predictTrain[initLength:], trainlabel[initLength:])
            CEBLS_TestSMAPE = getSMAPE(CEBLS_predictTest[initLength:], testlabel[initLength:])

            BLS_ESN_TrainSMAPE = getSMAPE(BLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            BLS_ESN_TestSMAPE = getSMAPE(BLS_ESN_predictTest[initLength:], testlabel[initLength:])

            CFBLS_ESN_TrainSMAPE = getSMAPE(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_ESN_TestSMAPE = getSMAPE(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_ESN_TrainSMAPE = getSMAPE(LCFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_ESN_TestSMAPE = getSMAPE(LCFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            # 计算R2
            BLS_TrainR2 = getR2(BLS_predictTrain[initLength:], trainlabel[initLength:])
            BLS_TestR2 = getR2(BLS_predictTest[initLength:], testlabel[initLength:])

            CFBLS_TrainR2 = getR2(CFBLS_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_TestR2 = getR2(CFBLS_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_TrainR2 = getR2(LCFBLS_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_TestR2 = getR2(LCFBLS_predictTest[initLength:], testlabel[initLength:])

            CEBLS_TrainR2 = getR2(CEBLS_predictTrain[initLength:], trainlabel[initLength:])
            CEBLS_TestR2 = getR2(CEBLS_predictTest[initLength:], testlabel[initLength:])

            BLS_ESN_TrainR2 = getR2(BLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            BLS_ESN_TestR2 = getR2(BLS_ESN_predictTest[initLength:], testlabel[initLength:])

            CFBLS_ESN_TrainR2 = getR2(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_ESN_TestR2 = getR2(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_ESN_TrainR2 = getR2(LCFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_ESN_TestR2 = getR2(LCFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            # 计算R
            BLS_TrainR = getR(BLS_predictTrain[initLength:], trainlabel[initLength:])
            BLS_TestR = getR(BLS_predictTest[initLength:], testlabel[initLength:])

            CFBLS_TrainR = getR(CFBLS_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_TestR = getR(CFBLS_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_TrainR = getR(LCFBLS_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_TestR = getR(LCFBLS_predictTest[initLength:], testlabel[initLength:])

            CEBLS_TrainR = getR(CEBLS_predictTrain[initLength:], trainlabel[initLength:])
            CEBLS_TestR = getR(CEBLS_predictTest[initLength:], testlabel[initLength:])

            BLS_ESN_TrainR = getR(BLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            BLS_ESN_TestR = getR(BLS_ESN_predictTest[initLength:], testlabel[initLength:])

            CFBLS_ESN_TrainR = getR(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            CFBLS_ESN_TestR = getR(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            LCFBLS_ESN_TrainR = getR(LCFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
            LCFBLS_ESN_TestR = getR(LCFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

            BLS_RMSE_TRAIN.append(BLS_TrainRMSE)
            BLS_RMSE_TEST.append(BLS_TestRMSE)
            CFBLS_RMSE_TRAIN.append(CFBLS_TrainRMSE)
            CFBLS_RMSE_TEST.append(CFBLS_TestRMSE)
            LCFBLS_RMSE_TRAIN.append(LCFBLS_TrainRMSE)
            LCFBLS_RMSE_TEST.append(LCFBLS_TestRMSE)
            CEBLS_RMSE_TRAIN.append(CEBLS_TrainRMSE)
            CEBLS_RMSE_TEST.append(CEBLS_TestRMSE)
            CFBLS_ESN_RMSE_TRAIN.append(CFBLS_ESN_TrainRMSE)
            CFBLS_ESN_RMSE_TEST.append(CFBLS_ESN_TestRMSE)
            BLS_ESN_RMSE_TRAIN.append(BLS_ESN_TrainRMSE)
            BLS_ESN_RMSE_TEST.append(BLS_ESN_TestRMSE)
            LCFBLS_ESN_RMSE_TRAIN.append(LCFBLS_ESN_TrainRMSE)
            LCFBLS_ESN_RMSE_TEST.append(LCFBLS_ESN_TestRMSE)

            BLS_MAE_TRAIN.append(BLS_TrainMAE)
            BLS_MAE_TEST.append(BLS_TestMAE)
            CFBLS_MAE_TRAIN.append(CFBLS_TrainMAE)
            CFBLS_MAE_TEST.append(CFBLS_TestMAE)
            LCFBLS_MAE_TRAIN.append(LCFBLS_TrainMAE)
            LCFBLS_MAE_TEST.append(LCFBLS_TestMAE)
            CEBLS_MAE_TRAIN.append(CEBLS_TrainMAE)
            CEBLS_MAE_TEST.append(CEBLS_TestMAE)
            CFBLS_ESN_MAE_TRAIN.append(CFBLS_ESN_TrainMAE)
            CFBLS_ESN_MAE_TEST.append(CFBLS_ESN_TestMAE)
            BLS_ESN_MAE_TRAIN.append(BLS_ESN_TrainMAE)
            BLS_ESN_MAE_TEST.append(BLS_ESN_TestMAE)
            LCFBLS_ESN_MAE_TRAIN.append(LCFBLS_ESN_TrainMAE)
            LCFBLS_ESN_MAE_TEST.append(LCFBLS_ESN_TestMAE)

            BLS_SMAPE_TRAIN.append(BLS_TrainSMAPE)
            BLS_SMAPE_TEST.append(BLS_TestSMAPE)
            CFBLS_SMAPE_TRAIN.append(CFBLS_TrainSMAPE)
            CFBLS_SMAPE_TEST.append(CFBLS_TestSMAPE)
            LCFBLS_SMAPE_TRAIN.append(LCFBLS_TrainSMAPE)
            LCFBLS_SMAPE_TEST.append(LCFBLS_TestSMAPE)
            CEBLS_SMAPE_TRAIN.append(CEBLS_TrainSMAPE)
            CEBLS_SMAPE_TEST.append(CEBLS_TestSMAPE)
            CFBLS_ESN_SMAPE_TRAIN.append(CFBLS_ESN_TrainSMAPE)
            CFBLS_ESN_SMAPE_TEST.append(CFBLS_ESN_TestSMAPE)
            BLS_ESN_SMAPE_TRAIN.append(BLS_ESN_TrainSMAPE)
            BLS_ESN_SMAPE_TEST.append(BLS_ESN_TestSMAPE)
            LCFBLS_ESN_SMAPE_TRAIN.append(LCFBLS_ESN_TrainSMAPE)
            LCFBLS_ESN_SMAPE_TEST.append(LCFBLS_ESN_TestSMAPE)

            BLS_R2_TRAIN.append(BLS_TrainR2)
            BLS_R2_TEST.append(BLS_TestR2)
            CFBLS_R2_TRAIN.append(CFBLS_TrainR2)
            CFBLS_R2_TEST.append(CFBLS_TestR2)
            LCFBLS_R2_TRAIN.append(LCFBLS_TrainR2)
            LCFBLS_R2_TEST.append(LCFBLS_TestR2)
            CEBLS_R2_TRAIN.append(CEBLS_TrainR2)
            CEBLS_R2_TEST.append(CEBLS_TestR2)
            CFBLS_ESN_R2_TRAIN.append(CFBLS_ESN_TrainR2)
            CFBLS_ESN_R2_TEST.append(CFBLS_ESN_TestR2)
            BLS_ESN_R2_TRAIN.append(BLS_ESN_TrainR2)
            BLS_ESN_R2_TEST.append(BLS_ESN_TestR2)
            LCFBLS_ESN_R2_TRAIN.append(LCFBLS_ESN_TrainR2)
            LCFBLS_ESN_R2_TEST.append(LCFBLS_ESN_TestR2)

            BLS_R_TRAIN.append(BLS_TrainR)
            BLS_R_TEST.append(BLS_TestR)
            CFBLS_R_TRAIN.append(CFBLS_TrainR)
            CFBLS_R_TEST.append(CFBLS_TestR)
            LCFBLS_R_TRAIN.append(LCFBLS_TrainR)
            LCFBLS_R_TEST.append(LCFBLS_TestR)
            CEBLS_R_TRAIN.append(CEBLS_TrainR)
            CEBLS_R_TEST.append(CEBLS_TestR)
            CFBLS_ESN_R_TRAIN.append(CFBLS_ESN_TrainR)
            CFBLS_ESN_R_TEST.append(CFBLS_ESN_TestR)
            BLS_ESN_R_TRAIN.append(BLS_ESN_TrainR)
            BLS_ESN_R_TEST.append(BLS_ESN_TestR)
            LCFBLS_ESN_R_TRAIN.append(LCFBLS_ESN_TrainR)
            LCFBLS_ESN_R_TEST.append(LCFBLS_ESN_TestR)

            BLS_predictTrain = np.array(BLS_predictTrain).T.reshape(-1, 1)
            BLS_predictTest = np.array(BLS_predictTest).T.reshape(-1, 1)
            CFBLS_predictTrain = np.array(CFBLS_predictTrain).T.reshape(-1, 1)
            CFBLS_predictTest = np.array(CFBLS_predictTest).T.reshape(-1, 1)
            LCFBLS_predictTrain = np.array(LCFBLS_predictTrain).T.reshape(-1, 1)
            LCFBLS_predictTest = np.array(LCFBLS_predictTest).T.reshape(-1, 1)
            CEBLS_predictTrain = np.array(CEBLS_predictTrain).T.reshape(-1, 1)
            CEBLS_predictTest = np.array(CEBLS_predictTest).T.reshape(-1, 1)
            BLS_ESN_predictTrain = np.array(BLS_ESN_predictTrain).T.reshape(-1, 1)
            BLS_ESN_predictTest = np.array(BLS_ESN_predictTest).T.reshape(-1, 1)
            CFBLS_ESN_predictTrain = np.array(CFBLS_ESN_predictTrain).T.reshape(-1, 1)
            CFBLS_ESN_predictTest = np.array(CFBLS_ESN_predictTest).T.reshape(-1, 1)
            LCFBLS_ESN_predictTrain = np.array(LCFBLS_ESN_predictTrain).T.reshape(-1, 1)
            LCFBLS_ESN_predictTest = np.array(LCFBLS_ESN_predictTest).T.reshape(-1, 1)

            print("-" * 100)
            print("BLS_TrainRMSE : {}, BLS_TrainR2 : {}".format(BLS_TrainRMSE, BLS_TrainR2))
            print("BLS_TestRMSE : {}, BLS_TestR2 : {}".format(BLS_TestRMSE, BLS_TestR2))
            print("CFBLS_TrainRMSE : {}, CFBLS_TrainR2 : {}".format(CFBLS_TrainRMSE, CFBLS_TrainR2))
            print("CFBLS_TestRMSE : {}, CFBLS_TestR2 : {}".format(CFBLS_TestRMSE, CFBLS_TestR2))
            print("LCFBLS_TrainRMSE : {}, LCFBLS_TrainR2 : {}".format(LCFBLS_TrainRMSE, LCFBLS_TrainR2))
            print("LCFBLS_TestRMSE : {}, LCFBLS_TestR2 : {}".format(LCFBLS_TestRMSE, LCFBLS_TestR2))
            print("CEBLS_TrainRMSE : {}, CEBLS_TrainR2 : {}".format(CEBLS_TrainRMSE, CEBLS_TrainR2))
            print("CEBLS_TestRMSE : {}, CEBLS_TestR2 : {}".format(CEBLS_TestRMSE, CEBLS_TestR2))
            print("BLS_ESN_TrainRMSE : {}, BLS_ESN_TrainR2 : {}".format(BLS_ESN_TrainRMSE, BLS_ESN_TrainR2))
            print("BLS_ESN_TestRMSE : {}, BLS_ESN_TestR2 : {}".format(BLS_ESN_TestRMSE, BLS_ESN_TestR2))
            print("CFBLS_ESN_TrainRMSE : {}, CFBLS_ESN_TrainR2 : {}".format(CFBLS_ESN_TrainRMSE, CFBLS_ESN_TrainR2))
            print("CFBLS_ESN_TestRMSE : {}, CFBLS_ESN_TestR2 : {}".format(CFBLS_ESN_TestRMSE, CFBLS_ESN_TestR2))
            print("LCFBLS_ESN_TrainRMSE : {}, LCFBLS_ESN_TrainR2 : {}".format(LCFBLS_ESN_TrainRMSE, LCFBLS_ESN_TrainR2))
            print("LCFBLS_ESN_TestRMSE : {}, LCFBLS_ESN_TestR2 : {}".format(LCFBLS_ESN_TestRMSE, LCFBLS_ESN_TestR2))

            #训练集的数据保存
            pd.DataFrame(BLS_predictTrain).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS\\train\\BLS_mapping_{}_enhance_{}.csv".format(i, j), index=False)
            pd.DataFrame(CFBLS_predictTrain).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS\\train\\CFBLS_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)
            pd.DataFrame(LCFBLS_predictTrain).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS\\train\\LCFBLS_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)
            pd.DataFrame(CEBLS_predictTrain).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CEBLS\\train\\CEBLS_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)

            pd.DataFrame(BLS_ESN_predictTrain).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS_ESN\\train\\BLS_ESN_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)
            pd.DataFrame(CFBLS_ESN_predictTrain).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\train\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)
            pd.DataFrame(LCFBLS_ESN_predictTrain).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS_ESN\\train\\LCFBLS_ESN_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)

            # 测试集的数据保存
            pd.DataFrame(BLS_predictTest).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)
            pd.DataFrame(CFBLS_predictTest).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)
            pd.DataFrame(LCFBLS_predictTest).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS\\test\\LCFBLS_mapping_{}_enhance_{}.csv".format(
                    i, j),
                index=False)
            pd.DataFrame(CEBLS_predictTest).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CEBLS\\test\\CEBLS_mapping_{}_enhance_{}.csv".format(i, j),
                index=False)

            pd.DataFrame(BLS_ESN_predictTest).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
                    i, j),
                index=False)
            pd.DataFrame(CFBLS_ESN_predictTest).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
                    i, j),
                index=False)
            pd.DataFrame(LCFBLS_ESN_predictTest).to_csv(
                "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS_ESN\\test\\LCFBLS_ESN_mapping_{}_enhance_{}.csv".format(
                    i, j),
                index=False)

            mappingNumberList.append(i)
            enhanceNumberList.append(j)

    print("BLS_TrainRMSE : ", BLS_RMSE_TRAIN)
    print("BLS_TestRMSE : ", BLS_RMSE_TEST)
    print("CFBLS_TrainRMSE : ", CFBLS_RMSE_TRAIN)
    print("CFBLS_TestRMSE : ", CFBLS_RMSE_TEST)
    print("LCFBLS_TrainRMSE : ", LCFBLS_RMSE_TRAIN)
    print("LCFBLS_TestRMSE : ", LCFBLS_RMSE_TEST)
    print("CEBLS_TrainRMSE : ", CEBLS_RMSE_TRAIN)
    print("CEBLS_TestRMSE : ", CEBLS_RMSE_TEST)
    print("BLS_ESN_TrainRMSE : ", BLS_ESN_RMSE_TRAIN)
    print("BLS_ESN_TestRMSE : ", BLS_ESN_RMSE_TEST)
    print("CFBLS_ESN_TrainRMSE : ", CFBLS_ESN_RMSE_TRAIN)
    print("CFBLS_ESN_TestRMSE : ", CFBLS_ESN_RMSE_TEST)
    print("LCFBLS_ESN_TrainRMSE : ", LCFBLS_ESN_RMSE_TRAIN)
    print("LCFBLS_ESN_TestRMSE : ", LCFBLS_ESN_RMSE_TEST)

    mappingNumberList = np.array(mappingNumberList)
    enhanceNumberList = np.array(enhanceNumberList)

    BLS_RMSE_TRAIN = np.array(BLS_RMSE_TRAIN)
    BLS_RMSE_TEST = np.array(BLS_RMSE_TEST)
    CFBLS_RMSE_TRAIN = np.array(CFBLS_RMSE_TRAIN)
    CFBLS_RMSE_TEST = np.array(CFBLS_RMSE_TEST)
    LCFBLS_RMSE_TRAIN = np.array(LCFBLS_RMSE_TRAIN)
    LCFBLS_RMSE_TEST = np.array(LCFBLS_RMSE_TEST)
    CEBLS_RMSE_TRAIN = np.array(CEBLS_RMSE_TRAIN)
    CEBLS_RMSE_TEST = np.array(CEBLS_RMSE_TEST)
    CFBLS_ESN_RMSE_TRAIN = np.array(CFBLS_ESN_RMSE_TRAIN)
    CFBLS_ESN_RMSE_TEST = np.array(CFBLS_ESN_RMSE_TEST)
    BLS_ESN_RMSE_TRAIN = np.array(BLS_ESN_RMSE_TRAIN)
    BLS_ESN_RMSE_TEST = np.array(BLS_ESN_RMSE_TEST)
    LCFBLS_ESN_RMSE_TRAIN = np.array(LCFBLS_ESN_RMSE_TRAIN)
    LCFBLS_ESN_RMSE_TEST = np.array(LCFBLS_ESN_RMSE_TEST)

    BLS_SMAPE_TRAIN = np.array(BLS_SMAPE_TRAIN)
    BLS_SMAPE_TEST = np.array(BLS_SMAPE_TEST)
    CFBLS_SMAPE_TRAIN = np.array(CFBLS_SMAPE_TRAIN)
    CFBLS_SMAPE_TEST = np.array(CFBLS_SMAPE_TEST)
    LCFBLS_SMAPE_TRAIN = np.array(LCFBLS_SMAPE_TRAIN)
    LCFBLS_SMAPE_TEST = np.array(LCFBLS_SMAPE_TEST)
    CEBLS_SMAPE_TRAIN = np.array(BLS_SMAPE_TRAIN)
    CEBLS_SMAPE_TEST = np.array(CEBLS_SMAPE_TRAIN)
    CFBLS_ESN_SMAPE_TRAIN = np.array(CFBLS_ESN_SMAPE_TRAIN)
    CFBLS_ESN_SMAPE_TEST = np.array(CFBLS_ESN_SMAPE_TEST)
    BLS_ESN_SMAPE_TRAIN = np.array(BLS_ESN_SMAPE_TRAIN)
    BLS_ESN_SMAPE_TEST = np.array(BLS_ESN_SMAPE_TEST)
    LCFBLS_ESN_SMAPE_TRAIN = np.array(LCFBLS_ESN_SMAPE_TRAIN)
    LCFBLS_ESN_SMAPE_TEST = np.array(LCFBLS_ESN_SMAPE_TEST)

    BLS_MAE_TRAIN = np.array(BLS_MAE_TRAIN)
    BLS_MAE_TEST = np.array(BLS_MAE_TEST)
    CFBLS_MAE_TRAIN = np.array(CFBLS_MAE_TRAIN)
    CFBLS_MAE_TEST = np.array(CFBLS_MAE_TEST)
    LCFBLS_MAE_TRAIN = np.array(LCFBLS_MAE_TRAIN)
    LCFBLS_MAE_TEST = np.array(LCFBLS_MAE_TEST)
    CEBLS_MAE_TRAIN = np.array(CEBLS_MAE_TRAIN)
    CEBLS_MAE_TEST = np.array(CEBLS_MAE_TEST)
    CFBLS_ESN_MAE_TRAIN = np.array(CFBLS_ESN_MAE_TRAIN)
    CFBLS_ESN_MAE_TEST = np.array(CFBLS_ESN_MAE_TEST)
    BLS_ESN_MAE_TRAIN = np.array(BLS_ESN_MAE_TRAIN)
    BLS_ESN_MAE_TEST = np.array(BLS_ESN_MAE_TEST)
    LCFBLS_ESN_MAE_TRAIN = np.array(LCFBLS_ESN_MAE_TRAIN)
    LCFBLS_ESN_MAE_TEST = np.array(LCFBLS_ESN_MAE_TEST)

    BLS_R2_TRAIN = np.array(BLS_R2_TRAIN)
    BLS_R2_TEST = np.array(BLS_R2_TEST)
    CFBLS_R2_TRAIN = np.array(CFBLS_R2_TRAIN)
    CFBLS_R2_TEST = np.array(CFBLS_R2_TEST)
    LCFBLS_R2_TRAIN = np.array(LCFBLS_R2_TRAIN)
    LCFBLS_R2_TEST = np.array(LCFBLS_R2_TEST)
    CEBLS_R2_TRAIN = np.array(CEBLS_R2_TRAIN)
    CEBLS_R2_TEST = np.array(CEBLS_R2_TEST)
    CFBLS_ESN_R2_TRAIN = np.array(CFBLS_ESN_R2_TRAIN)
    CFBLS_ESN_R2_TEST = np.array(CFBLS_ESN_R2_TEST)
    BLS_ESN_R2_TRAIN = np.array(BLS_ESN_R2_TRAIN)
    BLS_ESN_R2_TEST = np.array(BLS_ESN_R2_TEST)
    LCFBLS_ESN_R2_TRAIN = np.array(LCFBLS_ESN_R2_TRAIN)
    LCFBLS_ESN_R2_TEST = np.array(LCFBLS_ESN_R2_TEST)

    BLS_R_TRAIN = np.array(BLS_R_TRAIN)
    BLS_R_TEST = np.array(BLS_R_TEST)
    CFBLS_R_TRAIN = np.array(CFBLS_R_TRAIN)
    CFBLS_R_TEST = np.array(CFBLS_R_TEST)
    LCFBLS_R_TRAIN = np.array(LCFBLS_R_TRAIN)
    LCFBLS_R_TEST = np.array(LCFBLS_R_TEST)
    CEBLS_R_TRAIN = np.array(CEBLS_R_TRAIN)
    CEBLS_R_TEST = np.array(CEBLS_R_TEST)
    CFBLS_ESN_R_TRAIN = np.array(CFBLS_ESN_R_TRAIN)
    CFBLS_ESN_R_TEST = np.array(CFBLS_ESN_R_TEST)
    BLS_ESN_R_TRAIN = np.array(BLS_ESN_R_TRAIN)
    BLS_ESN_R_TEST = np.array(BLS_ESN_R_TEST)
    LCFBLS_ESN_R_TRAIN = np.array(LCFBLS_ESN_R_TRAIN)
    LCFBLS_ESN_R_TEST = np.array(LCFBLS_ESN_R_TEST)

    #时间处理
    BLS_TIME_TRAIN = np.array(BLS_TIME_TRAIN)
    BLS_TIME_TEST = np.array(BLS_TIME_TEST)
    CFBLS_TIME_TRAIN = np.array(CFBLS_TIME_TRAIN)
    CFBLS_TIME_TEST = np.array(CFBLS_TIME_TEST)
    LCFBLS_TIME_TRAIN = np.array(LCFBLS_TIME_TRAIN)
    LCFBLS_TIME_TEST = np.array(LCFBLS_TIME_TEST)
    CEBLS_TIME_TRAIN = np.array(CEBLS_TIME_TRAIN)
    CEBLS_TIME_TEST = np.array(CEBLS_TIME_TEST)
    BLS_ESN_TIME_TRAIN = np.array(BLS_ESN_TIME_TRAIN)
    BLS_ESN_TIME_TEST = np.array(BLS_ESN_TIME_TEST)
    CFBLS_ESN_TIME_TRAIN = np.array(CFBLS_ESN_TIME_TRAIN)
    CFBLS_ESN_TIME_TEST = np.array(CFBLS_ESN_TIME_TEST)
    LCFBLS_ESN_TIME_TRAIN = np.array(LCFBLS_ESN_TIME_TRAIN)
    LCFBLS_ESN_TIME_TEST = np.array(LCFBLS_ESN_TIME_TEST)

    #BLS的指标储存
    BLS_Train_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                          'enhanceNumber': enhanceNumberList.reshape(-1, ),
                          'RMSE': BLS_RMSE_TRAIN.reshape(-1, ), 'MAE': BLS_MAE_TRAIN.reshape(-1, ),
                          'SMAPE': BLS_SMAPE_TRAIN.reshape(-1, ), 'R2': BLS_R2_TRAIN.reshape(-1, ),
                          'R': BLS_R_TRAIN.reshape(-1, ), 'time': BLS_TIME_TRAIN.reshape(-1,),}
    pd.DataFrame(BLS_Train_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\BLS\\fangshan\\train\\BLS_train.csv', index=False)

    BLS_Test_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                          'enhanceNumber': enhanceNumberList.reshape(-1, ),
                          'RMSE': BLS_RMSE_TEST.reshape(-1, ), 'MAE': BLS_MAE_TEST.reshape(-1, ),
                          'SMAPE': BLS_SMAPE_TEST.reshape(-1, ), 'R2': BLS_R2_TEST.reshape(-1, ),
                          'R': BLS_R_TEST.reshape(-1, ), 'time': BLS_TIME_TEST.reshape(-1,),}
    pd.DataFrame(BLS_Test_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\BLS\\fangshan\\test\\BLS_test.csv', index=False)

    # CFBLS的指标储存
    CFBLS_Train_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                          'enhanceNumber': enhanceNumberList.reshape(-1, ),
                          'RMSE': CFBLS_RMSE_TRAIN.reshape(-1, ), 'MAE': CFBLS_MAE_TRAIN.reshape(-1, ),
                          'SMAPE': CFBLS_SMAPE_TRAIN.reshape(-1, ), 'R2': CFBLS_R2_TRAIN.reshape(-1, ),
                          'R': CFBLS_R_TRAIN.reshape(-1, ),'time': CFBLS_TIME_TRAIN.reshape(-1,), }
    pd.DataFrame(CFBLS_Train_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\CFBLS\\fangshan\\train\\CFBLS_train.csv', index=False)

    CFBLS_Test_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                         'enhanceNumber': enhanceNumberList.reshape(-1, ),
                         'RMSE': CFBLS_RMSE_TEST.reshape(-1, ), 'MAE': CFBLS_MAE_TEST.reshape(-1, ),
                         'SMAPE': CFBLS_SMAPE_TEST.reshape(-1, ), 'R2': CFBLS_R2_TEST.reshape(-1, ),
                         'R': CFBLS_R_TEST.reshape(-1, ), 'time': CFBLS_TIME_TEST.reshape(-1,),}
    pd.DataFrame(CFBLS_Test_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\CFBLS\\fangshan\\test\\CFBLS_test.csv', index=False)

    # LCFBLS的指标储存
    LCFBLS_Train_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                            'enhanceNumber': enhanceNumberList.reshape(-1, ),
                            'RMSE': LCFBLS_RMSE_TRAIN.reshape(-1, ), 'MAE': LCFBLS_MAE_TRAIN.reshape(-1, ),
                            'SMAPE': LCFBLS_SMAPE_TRAIN.reshape(-1, ), 'R2': LCFBLS_R2_TRAIN.reshape(-1, ),
                            'R': LCFBLS_R_TRAIN.reshape(-1, ), 'time': LCFBLS_TIME_TRAIN.reshape(-1,),}
    pd.DataFrame(LCFBLS_Train_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\LCFBLS\\fangshan\\train\\LCFBLS_train.csv', index=False)

    LCFBLS_Test_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                           'enhanceNumber': enhanceNumberList.reshape(-1, ),
                           'RMSE': LCFBLS_RMSE_TEST.reshape(-1, ), 'MAE': LCFBLS_MAE_TEST.reshape(-1, ),
                           'SMAPE': LCFBLS_SMAPE_TEST.reshape(-1, ), 'R2': LCFBLS_R2_TEST.reshape(-1, ),
                           'R': LCFBLS_R_TEST.reshape(-1, ), 'time': LCFBLS_TIME_TEST.reshape(-1,),}
    pd.DataFrame(LCFBLS_Test_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\LCFBLS\\fangshan\\test\\LCFBLS_test.csv', index=False)

    # CEBLS的指标储存
    CEBLS_Train_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                             'enhanceNumber': enhanceNumberList.reshape(-1, ),
                             'RMSE': CEBLS_RMSE_TRAIN.reshape(-1, ), 'MAE': CEBLS_MAE_TRAIN.reshape(-1, ),
                             'SMAPE': CEBLS_SMAPE_TRAIN.reshape(-1, ), 'R2': CEBLS_R2_TRAIN.reshape(-1, ),
                             'R': CEBLS_R_TRAIN.reshape(-1, ), 'time': CEBLS_TIME_TRAIN.reshape(-1,),}
    pd.DataFrame(CEBLS_Train_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\CEBLS\\fangshan\\train\\CEBLS_train.csv', index=False)

    CEBLS_Test_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                            'enhanceNumber': enhanceNumberList.reshape(-1, ),
                            'RMSE': CEBLS_RMSE_TEST.reshape(-1, ), 'MAE': CEBLS_MAE_TEST.reshape(-1, ),
                            'SMAPE': CEBLS_SMAPE_TEST.reshape(-1, ), 'R2': CEBLS_R2_TEST.reshape(-1, ),
                            'R': CEBLS_R_TEST.reshape(-1, ), 'time': CEBLS_TIME_TEST.reshape(-1,),}
    pd.DataFrame(CEBLS_Test_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\CEBLS\\fangshan\\test\\CEBLS_test.csv', index=False)

    # BLS_ESN的指标储存
    BLS_ESN_Train_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                             'enhanceNumber': enhanceNumberList.reshape(-1, ),
                             'RMSE': BLS_ESN_RMSE_TRAIN.reshape(-1, ), 'MAE': BLS_ESN_MAE_TRAIN.reshape(-1, ),
                             'SMAPE': BLS_ESN_SMAPE_TRAIN.reshape(-1, ), 'R2': BLS_ESN_R2_TRAIN.reshape(-1, ),
                             'R': BLS_ESN_R_TRAIN.reshape(-1, ), 'time': BLS_ESN_TIME_TRAIN.reshape(-1,),}
    pd.DataFrame(BLS_ESN_Train_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\BLS_ESN\\fangshan\\train\\BLS_ESN_train.csv', index=False)

    BLS_ESN_Test_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                            'enhanceNumber': enhanceNumberList.reshape(-1, ),
                            'RMSE': BLS_ESN_RMSE_TEST.reshape(-1, ), 'MAE': BLS_ESN_MAE_TEST.reshape(-1, ),
                            'SMAPE': BLS_ESN_SMAPE_TEST.reshape(-1, ), 'R2': BLS_ESN_R2_TEST.reshape(-1, ),
                            'R': BLS_ESN_R_TEST.reshape(-1, ), 'time': BLS_ESN_TIME_TEST.reshape(-1,),}
    pd.DataFrame(BLS_ESN_Test_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\BLS_ESN\\fangshan\\test\\BLS_ESN_test.csv', index=False)

    # CFBLS_ESN的指标储存
    CFBLS_ESN_Train_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                              'enhanceNumber': enhanceNumberList.reshape(-1, ),
                              'RMSE': CFBLS_ESN_RMSE_TRAIN.reshape(-1, ), 'MAE': CFBLS_ESN_MAE_TRAIN.reshape(-1, ),
                              'SMAPE': CFBLS_ESN_SMAPE_TRAIN.reshape(-1, ), 'R2': CFBLS_ESN_R2_TRAIN.reshape(-1, ),
                              'R': CFBLS_ESN_R_TRAIN.reshape(-1, ), 'time': CFBLS_ESN_TIME_TRAIN.reshape(-1,),}
    pd.DataFrame(CFBLS_ESN_Train_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\CFBLS_ESN\\fangshan\\train\\CFBLS_ESN_train.csv', index=False)

    CFBLS_ESN_Test_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                             'enhanceNumber': enhanceNumberList.reshape(-1, ),
                             'RMSE': CFBLS_ESN_RMSE_TEST.reshape(-1, ), 'MAE': CFBLS_ESN_MAE_TEST.reshape(-1, ),
                             'SMAPE': CFBLS_ESN_SMAPE_TEST.reshape(-1, ), 'R2': CFBLS_ESN_R2_TEST.reshape(-1, ),
                             'R': CFBLS_ESN_R_TEST.reshape(-1, ), 'time': CFBLS_ESN_TIME_TEST.reshape(-1,),}
    pd.DataFrame(CFBLS_ESN_Test_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\CFBLS_ESN\\fangshan\\test\\CFBLS_ESN_test.csv', index=False)

    # LCFBLS_ESN的指标储存
    LCFBLS_ESN_Train_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                                'enhanceNumber': enhanceNumberList.reshape(-1, ),
                                'RMSE': LCFBLS_ESN_RMSE_TRAIN.reshape(-1, ), 'MAE': LCFBLS_ESN_MAE_TRAIN.reshape(-1, ),
                                'SMAPE': LCFBLS_ESN_SMAPE_TRAIN.reshape(-1, ), 'R2': LCFBLS_ESN_R2_TRAIN.reshape(-1, ),
                                'R': LCFBLS_ESN_R_TRAIN.reshape(-1, ),'time': LCFBLS_ESN_TIME_TRAIN.reshape(-1,),}
    pd.DataFrame(LCFBLS_ESN_Train_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\LCFBLS_ESN\\fangshan\\train\\LCFBLS_ESN_train.csv', index=False)

    LCFBLS_ESN_Test_datafram = {'mappingNumber': mappingNumberList.reshape(-1, ),
                               'enhanceNumber': enhanceNumberList.reshape(-1, ),
                               'RMSE': LCFBLS_ESN_RMSE_TEST.reshape(-1, ), 'MAE': LCFBLS_ESN_MAE_TEST.reshape(-1, ),
                               'SMAPE': LCFBLS_ESN_SMAPE_TEST.reshape(-1, ), 'R2': LCFBLS_ESN_R2_TEST.reshape(-1, ),
                               'R': LCFBLS_ESN_R_TEST.reshape(-1, ), 'time':LCFBLS_ESN_TIME_TEST.reshape(-1,),}
    pd.DataFrame(LCFBLS_ESN_Test_datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\LCFBLS_ESN\\fangshan\\test\\LCFBLS_ESN_test.csv', index=False)

    #所有模型的所有指标来个汇总吧
    dataframTest = {'mappingNumber': mappingNumberList.reshape(-1, ),
                    'enhanceNumber': enhanceNumberList.reshape(-1, ),
                    'BLS_RMSE': BLS_RMSE_TEST.reshape(-1, ), 'BLS_MAE': BLS_MAE_TEST.reshape(-1, ),
                    'BLS_SMAPE': BLS_SMAPE_TEST.reshape(-1, ), 'BLS_R2': BLS_R2_TEST.reshape(-1, ),
                    'BLS_R': BLS_R_TEST.reshape(-1, ),'BLS_TIME': BLS_TIME_TEST.reshape(-1,),
                     'CFBLS_RMSE': CFBLS_RMSE_TEST.reshape(-1, ), 'CFBLS_MAE': CFBLS_MAE_TEST.reshape(-1, ),
                     'CFBLS_SMAPE': CFBLS_SMAPE_TEST.reshape(-1, ), 'CFBLS_R2': CFBLS_R2_TEST.reshape(-1, ),
                     'CFBLS_R': CFBLS_R_TEST.reshape(-1, ),'CFBLS_TIME': CFBLS_TIME_TEST.reshape(-1,),
                     'LCFBLS_RMSE': LCFBLS_RMSE_TEST.reshape(-1, ), 'LCFBLS_MAE': LCFBLS_MAE_TEST.reshape(-1, ),
                     'LCFBLS_SMAPE': LCFBLS_SMAPE_TEST.reshape(-1, ), 'LCFBLS_R2': LCFBLS_R2_TEST.reshape(-1, ),
                     'LCFBLS_R': LCFBLS_R_TEST.reshape(-1, ),'LCFBLS_TIME': LCFBLS_TIME_TEST.reshape(-1,),
                     'CEBLS_RMSE': CEBLS_RMSE_TEST.reshape(-1, ), 'CEBLS_MAE': CEBLS_MAE_TEST.reshape(-1, ),
                     'CEBLS_SMAPE': CEBLS_SMAPE_TEST.reshape(-1, ), 'CEBLS_R2': CEBLS_R2_TEST.reshape(-1, ),
                     'CEBLS_R': CEBLS_R_TEST.reshape(-1, ),'CEBLS_TIME': CEBLS_TIME_TEST.reshape(-1,),
                     'BLS_ESN_RMSE': BLS_ESN_RMSE_TEST.reshape(-1, ), 'BLS_ESN_MAE': BLS_ESN_MAE_TEST.reshape(-1, ),
                     'BLS_ESN_SMAPE': BLS_ESN_SMAPE_TEST.reshape(-1, ), 'BLS_ESN_R2': BLS_ESN_R2_TEST.reshape(-1, ),
                     'BLS_ESN_R': BLS_ESN_R_TEST.reshape(-1, ),'BLS_ESN_TIME': BLS_ESN_TIME_TEST.reshape(-1,),
                     'CFBLS_ESN_RMSE': CFBLS_ESN_RMSE_TEST.reshape(-1, ), 'CFBLS_ESN_MAE': CFBLS_ESN_MAE_TEST.reshape(-1, ),
                     'CFBLS_ESN_SMAPE': CFBLS_ESN_SMAPE_TEST.reshape(-1, ), 'CFBLS_ESN_R2': CFBLS_ESN_R2_TEST.reshape(-1, ),
                     'CFBLS_ESN_R': CFBLS_ESN_R_TEST.reshape(-1, ),'CFBLS_ESN_TIME': CFBLS_ESN_TIME_TEST.reshape(-1,),
                     'LCFBLS_ESN_RMSE': LCFBLS_ESN_RMSE_TEST.reshape(-1, ), 'LCFBLS_ESN_MAE': LCFBLS_ESN_MAE_TEST.reshape(-1, ),
                     'LCFBLS_ESN_SMAPE': LCFBLS_ESN_SMAPE_TEST.reshape(-1, ),'LCFBLS_ESN_R2': LCFBLS_ESN_R2_TEST.reshape(-1, ),
                     'LCFBLS_ESN_R': LCFBLS_ESN_R_TEST.reshape(-1, ),'LCFBLS_ESN_TIME': LCFBLS_ESN_TIME_TEST.reshape(-1,),
                     }
    pd.DataFrame(dataframTest).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\huizong_test.csv', index=False)

    dataframTrain = {'mappingNumber': mappingNumberList.reshape(-1, ),
                    'enhanceNumber': enhanceNumberList.reshape(-1, ),
                    'BLS_RMSE': BLS_RMSE_TRAIN.reshape(-1, ), 'BLS_MAE': BLS_MAE_TRAIN.reshape(-1, ),
                    'BLS_SMAPE': BLS_SMAPE_TRAIN.reshape(-1, ), 'BLS_R2': BLS_R2_TRAIN.reshape(-1, ),
                    'BLS_R': BLS_R_TRAIN.reshape(-1, ),'BLS_TIME': BLS_TIME_TRAIN.reshape(-1,),
                    'CFBLS_RMSE': CFBLS_RMSE_TRAIN.reshape(-1, ), 'CFBLS_MAE': CFBLS_MAE_TRAIN.reshape(-1, ),
                    'CFBLS_SMAPE': CFBLS_SMAPE_TRAIN.reshape(-1, ), 'CFBLS_R2': CFBLS_R2_TRAIN.reshape(-1, ),
                    'CFBLS_R': CFBLS_R_TRAIN.reshape(-1, ),'CFBLS_TIME': CFBLS_TIME_TRAIN.reshape(-1,),
                    'LCFBLS_RMSE': LCFBLS_RMSE_TRAIN.reshape(-1, ), 'LCFBLS_MAE': LCFBLS_MAE_TRAIN.reshape(-1, ),
                    'LCFBLS_SMAPE': LCFBLS_SMAPE_TRAIN.reshape(-1, ), 'LCFBLS_R2': LCFBLS_R2_TRAIN.reshape(-1, ),
                    'LCFBLS_R': LCFBLS_R_TRAIN.reshape(-1, ),'LCFBLS_TIME': LCFBLS_TIME_TRAIN.reshape(-1,),
                    'CEBLS_RMSE': CEBLS_RMSE_TRAIN.reshape(-1, ), 'CEBLS_MAE': CEBLS_MAE_TRAIN.reshape(-1, ),
                    'CEBLS_SMAPE': CEBLS_SMAPE_TRAIN.reshape(-1, ), 'CEBLS_R2': CEBLS_R2_TRAIN.reshape(-1, ),
                    'CEBLS_R': CEBLS_R_TRAIN.reshape(-1, ),'CEBLS_TIME': CEBLS_TIME_TRAIN.reshape(-1,),
                    'BLS_ESN_RMSE': BLS_ESN_RMSE_TRAIN.reshape(-1, ), 'BLS_ESN_MAE': BLS_ESN_MAE_TRAIN.reshape(-1, ),
                    'BLS_ESN_SMAPE': BLS_ESN_SMAPE_TRAIN.reshape(-1, ), 'BLS_ESN_R2': BLS_ESN_R2_TRAIN.reshape(-1, ),
                    'BLS_ESN_R': BLS_ESN_R_TRAIN.reshape(-1, ),'BLS_ESN_TIME': BLS_ESN_TIME_TRAIN.reshape(-1,),
                    'CFBLS_ESN_RMSE': CFBLS_ESN_RMSE_TRAIN.reshape(-1, ),
                    'CFBLS_ESN_MAE': CFBLS_ESN_MAE_TRAIN.reshape(-1, ),
                    'CFBLS_ESN_SMAPE': CFBLS_ESN_SMAPE_TRAIN.reshape(-1, ),
                    'CFBLS_ESN_R2': CFBLS_ESN_R2_TRAIN.reshape(-1, ),
                    'CFBLS_ESN_R': CFBLS_ESN_R_TRAIN.reshape(-1, ),
                     'CFBLS_ESN_TIME': CFBLS_ESN_TIME_TRAIN.reshape(-1, ),
                    'LCFBLS_ESN_RMSE': LCFBLS_ESN_RMSE_TRAIN.reshape(-1, ),
                    'LCFBLS_ESN_MAE': LCFBLS_ESN_MAE_TRAIN.reshape(-1, ),
                    'LCFBLS_ESN_SMAPE': LCFBLS_ESN_SMAPE_TRAIN.reshape(-1, ),
                    'LCFBLS_ESN_R2': LCFBLS_ESN_R2_TRAIN.reshape(-1, ),
                    'LCFBLS_ESN_R': LCFBLS_ESN_R_TRAIN.reshape(-1, ),
                     'LCFBLS_ESN_TIME': LCFBLS_ESN_TIME_TRAIN.reshape(-1, ),
                    }
    pd.DataFrame(dataframTrain).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\huizong_train.csv', index=False)

    dataframTrain_RMSE_TIME = {'mappingNumber': mappingNumberList.reshape(-1, ),
                     'enhanceNumber': enhanceNumberList.reshape(-1, ),
                     'BLS_RMSE': BLS_RMSE_TRAIN.reshape(-1, ),  'BLS_R2': BLS_R2_TRAIN.reshape(-1, ),
                    'BLS_TIME': BLS_TIME_TRAIN.reshape(-1, ),
                     'CFBLS_RMSE': CFBLS_RMSE_TRAIN.reshape(-1, ), 'CFBLS_R2': CFBLS_R2_TRAIN.reshape(-1, ),
                     'CFBLS_TIME': CFBLS_TIME_TRAIN.reshape(-1, ),
                     'LCFBLS_RMSE': LCFBLS_RMSE_TRAIN.reshape(-1, ),  'LCFBLS_R2': LCFBLS_R2_TRAIN.reshape(-1, ),
                      'LCFBLS_TIME': LCFBLS_TIME_TRAIN.reshape(-1, ),
                     'CEBLS_RMSE': CEBLS_RMSE_TRAIN.reshape(-1, ),  'CEBLS_R2': CEBLS_R2_TRAIN.reshape(-1, ),
                     'CEBLS_TIME': CEBLS_TIME_TRAIN.reshape(-1, ),
                     'BLS_ESN_RMSE': BLS_ESN_RMSE_TRAIN.reshape(-1, ),  'BLS_ESN_R2': BLS_ESN_R2_TRAIN.reshape(-1, ),
                     'BLS_ESN_TIME': BLS_ESN_TIME_TRAIN.reshape(-1, ),
                     'CFBLS_ESN_RMSE': CFBLS_ESN_RMSE_TRAIN.reshape(-1, ),
                     'CFBLS_ESN_R2': CFBLS_ESN_R2_TRAIN.reshape(-1, ),
                     'CFBLS_ESN_TIME': CFBLS_ESN_TIME_TRAIN.reshape(-1, ),
                     'LCFBLS_ESN_RMSE': LCFBLS_ESN_RMSE_TRAIN.reshape(-1, ),
                     'LCFBLS_ESN_R2': LCFBLS_ESN_R2_TRAIN.reshape(-1, ),
                     'LCFBLS_ESN_TIME': LCFBLS_ESN_TIME_TRAIN.reshape(-1, ),
                     }
    pd.DataFrame(dataframTrain_RMSE_TIME).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\huizong_train_RMSE_TIME.csv', index=False)

    dataframTest_RMSE_TIME = {'mappingNumber': mappingNumberList.reshape(-1, ),
                               'enhanceNumber': enhanceNumberList.reshape(-1, ),
                               'BLS_RMSE': BLS_RMSE_TEST.reshape(-1, ), 'BLS_R2': BLS_R2_TEST.reshape(-1, ),
                               'BLS_TIME': BLS_TIME_TEST.reshape(-1, ),
                               'CFBLS_RMSE': CFBLS_RMSE_TEST.reshape(-1, ), 'CFBLS_R2': CFBLS_R2_TEST.reshape(-1, ),
                               'CFBLS_TIME': CFBLS_TIME_TEST.reshape(-1, ),
                               'LCFBLS_RMSE': LCFBLS_RMSE_TEST.reshape(-1, ),
                               'LCFBLS_R2': LCFBLS_R2_TEST.reshape(-1, ),
                               'LCFBLS_TIME': LCFBLS_TIME_TEST.reshape(-1, ),
                               'CEBLS_RMSE': CEBLS_RMSE_TEST.reshape(-1, ), 'CEBLS_R2': CEBLS_R2_TEST.reshape(-1, ),
                               'CEBLS_TIME': CEBLS_TIME_TEST.reshape(-1, ),
                               'BLS_ESN_RMSE': BLS_ESN_RMSE_TEST.reshape(-1, ),
                               'BLS_ESN_R2': BLS_ESN_R2_TEST.reshape(-1, ),
                               'BLS_ESN_TIME': BLS_ESN_TIME_TEST.reshape(-1, ),
                               'CFBLS_ESN_RMSE': CFBLS_ESN_RMSE_TEST.reshape(-1, ),
                               'CFBLS_ESN_R2': CFBLS_ESN_R2_TEST.reshape(-1, ),
                               'CFBLS_ESN_TIME': CFBLS_ESN_TIME_TEST.reshape(-1, ),
                               'LCFBLS_ESN_RMSE': LCFBLS_ESN_RMSE_TEST.reshape(-1, ),
                               'LCFBLS_ESN_R2': LCFBLS_ESN_R2_TEST.reshape(-1, ),
                               'LCFBLS_ESN_TIME': LCFBLS_ESN_TIME_TEST.reshape(-1, ),
                               }
    pd.DataFrame(dataframTest_RMSE_TIME).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\huizong_test_RMSE_TIME.csv', index=False)

# fig = plt.figure(figsize=(13, 5))
    # # x = np.arange(len(testlabel)-initLength)
    # x = np.arange(200)
    # plt.plot(x, testlabel[initLength:initLength+200], color='#FF0000', label='real')
    # # plt.plot(x, BLS_predictTest[:600], color='#00B0F0', label='BLS')
    # plt.plot(x, CFBLS_predictTest[initLength:initLength+200], color='#CD853F', label='CFBLS')
    # plt.plot(x, LCFBLS_predictTest[initLength:initLength+200], color='#00B050', label='LCFBLS')
    # # plt.plot(x, CEBLS_predictTest[:600], color='#92D050', label='CEBLS')
    # plt.plot(x, LCFBLS_ESN_predictTest[initLength:initLength+200], color='#92D050', label='LCFBLS_ESN')
    # plt.legend(loc='upper left')  # 把图例设置在外边
    # plt.ylabel('AQI')
    # plt.show()
