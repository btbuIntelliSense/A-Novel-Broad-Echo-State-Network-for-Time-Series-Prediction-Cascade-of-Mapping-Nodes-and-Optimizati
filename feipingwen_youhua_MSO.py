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
from statsmodels.tsa.stattools import adfuller

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

name = 'MSO_8'

def data_process_slide(X, Y, X_sample_size, Y__sample_size, slide_size,iteration):
    OUT = []
    y = []
    X.reshape(-1, 1)
    Y.reshape(-1, 1)
    for i in range(iteration):
        OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
        y.append(Y[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
    OUT = np.array(OUT).reshape(iteration,X_sample_size)
    y = np.array(y).reshape(iteration,Y__sample_size)
    print("OUT.shape : ", OUT.shape)
    print("y.shape : ", y.shape)
    return OUT, y

def generate_data_by_slide( X_sample_size, Y__sample_size, slide_size):
    '''
    前提是原始数据是一列数据，现在把它划分为（iteration,X_sample_size + Y__sample_size）
    :param X_sample_size: X的样本个数
    :param Y__sample_size: Y的样本个数
    :param slide_size: 窗口的滑动步长
    :return: None
    '''
    path = 'E:\yan_2\CFBLS_LCFBLS复现\dataset\MSO_data_8.csv'
    X = pd.read_csv(path)

    OUT = []
    y = []

    iteration = (len(X) - X_sample_size - Y__sample_size)//slide_size
    for i in range(iteration):
        OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
        y.append(X[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
    OUT = np.array(OUT).reshape(iteration,X_sample_size)
    y = np.array(y).reshape(iteration,Y__sample_size)
    print("OUT.shape : ", OUT.shape)
    print("y.shape : ", y.shape)
    data = np.hstack((OUT,y))

    pd.DataFrame(data).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name,slide_size),
                             index=False)

def getDataFromCSV(path,Xcolumns,Ycolumns):
    '''
    将数据切片
    :return: 没有归一化的数据
    '''
    data = pd.read_csv(path).values[1:15000, : ]
    trainLength = int(data.shape[0] * 0.8)

    traindata = data[:trainLength, :Xcolumns]
    trainlabel = data[:trainLength, Xcolumns:Xcolumns+Ycolumns]
    testdata = data[trainLength:, :Xcolumns]
    testlabel = data[trainLength:, Xcolumns:Xcolumns+Ycolumns]

    pd.DataFrame(testlabel.reshape(-1, 1)).to_csv(
        "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv", index=False)
    pd.DataFrame(trainlabel.reshape(-1, 1)).to_csv(
        "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\train\\trainReal.csv", index=False)

    print("traindata.shape : ",traindata.shape)
    print("trainlabel.shape : ", trainlabel.shape)
    print("testdata.shape : ", testdata.shape)
    print("testlabel.shape : ", testlabel.shape)

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

CFBLS_ESN_RMSE_TEST = []
CFBLS_ESN_MAE_TEST = []
CFBLS_ESN_SMAPE_TEST = []
CFBLS_ESN_R2_TEST = []
CFBLS_ESN_TIME_TEST = []
CFBLS_ESN_TIME_TRAIN = []
CFBLS_ESN_Regularization_coefficient = []
CFBLS_ESN_M = []
CFBLS_ESN_S = []
CFBLS_ESN_P_ADF = []
CFBLS_ESN_1_ADF = []
CFBLS_ESN_5_ADF = []
CFBLS_ESN_10_ADF = []

Regularization_factor_real = 0.0
S1 = 0.0
S2 = 0.0
M = 0.0

if __name__ == '__main__':
    X_sample_size = 12
    Y__sample_size = 1
    slide_size = 1

    generate_data_by_slide(X_sample_size, Y__sample_size, slide_size)

    path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name, slide_size)
    traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
    initLength = 100

    P_ADF_real_data = adfuller(np.array(testlabel[1000:1120, :]), regression='ctt', autolag='AIC')

    c = 3**-20

    for i in range(1, 100, 1):


        # scaler1 = MinMaxScaler()
        # scaler2 = MinMaxScaler()
        # scaler3 = MinMaxScaler()
        # scaler4 = MinMaxScaler()
        #
        # traindata = scaler1.fit_transform(traindata)
        # trainlabel = scaler2.fit_transform(trainlabel)
        # testdata = scaler3.fit_transform(testdata)
        # testlabel = scaler4.fit_transform(testlabel)



        # N1是映射层每个窗口神经元的个数，N2是映射层窗口个数，N3是强化层神经元个数（ESN的个数）
        CFBLS_ESN_predictTrain, CFBLS_ESN_predictTest, CFBLS_ESN_TRAIN_TIME, CFBLS_ESN_TEST_TIME = CFBLS_ESN(traindata,
                                                                                                             trainlabel,
                                                                                                             testdata,
                                                                                                             testlabel,
                                                                                                             s=0.8,
                                                                                                             c= c,
                                                                                                             N1=10,
                                                                                                             N2=36,
                                                                                                             N3=18)

        CFBLS_ESN_Regularization_coefficient.append(c)

        # CFBLS_ESN_predictTrain = scaler2.inverse_transform(CFBLS_ESN_predictTrain)
        # CFBLS_ESN_predictTest = scaler4.inverse_transform(CFBLS_ESN_predictTest)
        #
        # traindata = scaler1.inverse_transform(traindata)
        # testdata = scaler3.inverse_transform(testdata)
        # trainlabel = scaler2.inverse_transform(trainlabel)
        # testlabel = scaler4.inverse_transform(testlabel)

        CFBLS_ESN_TrainRMSE = getRMSE(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
        CFBLS_ESN_TestRMSE = getRMSE(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

        CFBLS_ESN_TrainR2 = getR2(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
        CFBLS_ESN_TestR2 = getR2(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

        CFBLS_ESN_TrainMAE = getMAE(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
        CFBLS_ESN_TestMAE = getMAE(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

        CFBLS_ESN_TrainSMAPE = getSMAPE(CFBLS_ESN_predictTrain[initLength:], trainlabel[initLength:])
        CFBLS_ESN_TestSMAPE = getSMAPE(CFBLS_ESN_predictTest[initLength:], testlabel[initLength:])

        pd.DataFrame(CFBLS_ESN_predictTrain).to_csv(
            "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\自平稳优化指标体系实验结果数据\\调试版\\train\\train_{}_补.csv".format(i),
            index=False)
        pd.DataFrame(CFBLS_ESN_predictTest).to_csv(
            "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\自平稳优化指标体系实验结果数据\\调试版\\test\\test_{}_补.csv".format(i),
            index=False)

        print("-------------------------------Regularization factor : {} -------------------------------".format(c))
        if i == 1:
            predict_data_P_ADF = adfuller(np.array(testlabel[1000:1120,:]), regression='ctt', autolag='AIC')
            S1 = abs(P_ADF_real_data[1] - predict_data_P_ADF[1]) * CFBLS_ESN_TestRMSE

            CFBLS_ESN_P_ADF.append(predict_data_P_ADF[1])
            CFBLS_ESN_1_ADF.append(predict_data_P_ADF[4]['1%'])
            CFBLS_ESN_5_ADF.append(predict_data_P_ADF[4]['5%'])
            CFBLS_ESN_10_ADF.append(predict_data_P_ADF[4]['10%'])

            print("S1 : {}, S2 : {}, M : {}, C : {}, RMSE : {}".format(S1, S2, M, c, CFBLS_ESN_TestRMSE))

        if i >= 2:
            predict_data_P_ADF = adfuller(np.array(testlabel[1000:1120, :]), regression='ctt', autolag='AIC')
            S2 = abs(predict_data_P_ADF[1] - P_ADF_real_data[1]) * CFBLS_ESN_TestRMSE
            M = S2 / S1

            L = sum(CFBLS_ESN_predictTest[initLength:] - testlabel[initLength:])

            if L >= 0:
                c = 1.1 * c
            else:
                c = 0.9 * c

            print("次数 : {}  S1 : {}, S2 : {}, M : {}, C : {}, RMSE : {}".format(i, S1, S2, M, c, CFBLS_ESN_TestRMSE))
            S1 = S2

            CFBLS_ESN_P_ADF.append(predict_data_P_ADF[1])
            CFBLS_ESN_1_ADF.append(predict_data_P_ADF[4]['1%'])
            CFBLS_ESN_5_ADF.append(predict_data_P_ADF[4]['5%'])
            CFBLS_ESN_10_ADF.append(predict_data_P_ADF[4]['10%'])

        CFBLS_ESN_RMSE_TEST.append(CFBLS_ESN_TestRMSE)
        CFBLS_ESN_MAE_TEST.append(CFBLS_ESN_TestMAE)
        CFBLS_ESN_SMAPE_TEST.append(CFBLS_ESN_TestSMAPE)
        CFBLS_ESN_R2_TEST.append(CFBLS_ESN_TestR2)
        CFBLS_ESN_TIME_TEST.append(CFBLS_ESN_TEST_TIME)
        CFBLS_ESN_TIME_TRAIN.append(CFBLS_ESN_TRAIN_TIME)
        CFBLS_ESN_M.append(M)
        CFBLS_ESN_S.append(S1)

    CFBLS_ESN_RMSE_TEST = np.array(CFBLS_ESN_RMSE_TEST)
    CFBLS_ESN_MAE_TEST = np.array(CFBLS_ESN_MAE_TEST)
    CFBLS_ESN_SMAPE_TEST = np.array(CFBLS_ESN_SMAPE_TEST)
    CFBLS_ESN_R2_TEST = np.array(CFBLS_ESN_R2_TEST)
    CFBLS_ESN_TIME_TEST = np.array(CFBLS_ESN_TIME_TEST)
    CFBLS_ESN_TIME_TRAIN = np.array(CFBLS_ESN_TIME_TRAIN)
    CFBLS_ESN_M = np.array(CFBLS_ESN_M)
    CFBLS_ESN_S = np.array(CFBLS_ESN_S)
    CFBLS_ESN_P_ADF = np.array(CFBLS_ESN_P_ADF)
    CFBLS_ESN_1_ADF = np.array(CFBLS_ESN_1_ADF)
    CFBLS_ESN_5_ADF = np.array(CFBLS_ESN_5_ADF)
    CFBLS_ESN_10_ADF = np.array(CFBLS_ESN_10_ADF)
    CFBLS_ESN_Regularization_coefficient = np.array(CFBLS_ESN_Regularization_coefficient)

    dataframTest = {'RMSE':CFBLS_ESN_RMSE_TEST.reshape(-1, ),
                    'MAE':CFBLS_ESN_MAE_TEST.reshape(-1, ),
                    'SMAPE':CFBLS_ESN_SMAPE_TEST.reshape(-1, ),
                    'R2':CFBLS_ESN_R2_TEST.reshape(-1, ),
                    'test_time':CFBLS_ESN_TIME_TEST.reshape(-1, ),
                    'train_time':CFBLS_ESN_TIME_TRAIN.reshape(-1, ),
                    'Regularization_coefficient': CFBLS_ESN_Regularization_coefficient.reshape(-1, ),
                    'P': CFBLS_ESN_P_ADF.reshape(-1, ),
                    'M':CFBLS_ESN_M.reshape(-1, ),
                    'S':CFBLS_ESN_S.reshape(-1, ),
                    '_1_':CFBLS_ESN_1_ADF.reshape(-1, ),
                    '_5_':CFBLS_ESN_5_ADF.reshape(-1, ),
                    '_10_':CFBLS_ESN_10_ADF.reshape(-1, ),}
    pd.DataFrame(dataframTest).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\MSO_8\\自平稳优化指标体系数据汇总-调试版_补.csv', index=False)









