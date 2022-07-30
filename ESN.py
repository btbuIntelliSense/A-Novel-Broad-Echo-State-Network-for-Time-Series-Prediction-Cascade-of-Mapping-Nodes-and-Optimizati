# -*- coding: utf-8 -*-
"""
Echo State Networks
http://minds.jacobs-university.de/mantas
http://organic.elis.ugent.be/
http://www.scholarpedia.org/article/Echo_state_network
Mantas Lukoševičius "A Practical Guide to Applying Echo State Networks"
Konrad Stanek "Reservoir computing in financial forecasting with committee methods"
"""
import numpy as np
from sklearn import preprocessing
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import datetime
from numpy import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

name = "MSO"
np.random.seed(42)


RMSEList = []
MAEList = []
SMAPEList = []
R2List = []
RList = []
resSizeNumberList = []
trainTimeList = []
testTimeList = []

def data_process(X, Y, know_num, pre_num, iteration):
    '''
    X:输入数据 shape 为 (len, ) X = np.array(data.loc[:]['0.1']).reshape(-1, 1) 再将X归一化
    Y:对应的Y shape 为 (len, ) Y同上
    know_num:已知个数
    pre_num:预测个数
    iteration:迭代次数
    OUT：know_num * iteration 已知个数 * 迭代次数
    实际使用输入数据个数：iteration * pre_num + (know_num-pre_num) 迭代次数 * 预测个数 + （已知个数-预测个数）
    '''
    OUT = []
    X.reshape(-1, 1)
    Y.reshape(-1, 1)
    for i in range(iteration):  # 训练样本个数iteration ， 一个样本里边有know_num多个数据
        for j in range(know_num):  # know_num 是一个样本里的训练数量
            OUT.append(X[i * pre_num + j])  # 延时t个数
    len_in = iteration * pre_num + (know_num - pre_num)  # 实际使用输入数据个数
    len_y = iteration * pre_num  # 对应 y 的长度
    len_x = know_num * iteration
    y = Y[know_num:know_num + len_y].reshape(-1, pre_num)
    x = np.array(OUT).reshape(-1, know_num, 1)
    print("输出X长度：{} 输出Y长度：{} 共使用X {} 个".format(len_x, len_y, len_in))
    return x, y


# #房山数据集
# def getDataFromCSV(path):
#     '''
#     将数据切片
#     :return: 没有归一化的数据
#     '''
#     data = pd.read_csv(path).values[1000:16000, :].reshape(15000, 7)
#
#     initLen = 0
#
#     label = data[:, :1].reshape(-1, 1)
#     data = data[:, 1: 7]
#     # print(data.shape, max(label) + 1)
#     # traindata , testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.01,random_state = 0)
#
#     trainLen = 12000
#     testLen = len(data) - trainLen
#     traindata = data[initLen:trainLen, :]
#     trainlabel = label[initLen:trainLen]
#     testdata = data[trainLen: trainLen + testLen, :]
#     testlabel = label[trainLen: trainLen + testLen]
#     return traindata, trainlabel, testdata, testlabel



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
    name = 'MSO_8'
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
# #MSO_8数据集
# def getDataFromCSV(path,Xcolumns,Ycolumns):
#     '''
#     将数据切片
#     :return: 没有归一化的数据
#     '''
#     data = pd.read_csv(path).values[1:15000, : ]
#     trainLength = int(data.shape[0] * 0.8)
#
#     traindata = data[:trainLength, :Xcolumns]
#     trainlabel = data[:trainLength, Xcolumns:Xcolumns+Ycolumns]
#     testdata = data[trainLength:, :Xcolumns]
#     testlabel = data[trainLength:, Xcolumns:Xcolumns+Ycolumns]
#
#     pd.DataFrame(testlabel.reshape(-1, 1)).to_csv(
#         "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv", index=False)
#     pd.DataFrame(trainlabel.reshape(-1, 1)).to_csv(
#         "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\train\\trainReal.csv", index=False)
#
#     print("traindata.shape : ",traindata.shape)
#     print("trainlabel.shape : ", trainlabel.shape)
#     print("testdata.shape : ", testdata.shape)
#     print("testlabel.shape : ", testlabel.shape)
#
#     return traindata, trainlabel, testdata, testlabel

#Sunsplot数据集
def getDataFromCSV(path,Xcolumns,Ycolumns):
    '''
    将数据切片
    :return: 没有归一化的数据
    '''
    data = pd.read_csv(path).values[1:2401, : ]
    trainLength = int(data.shape[0] * 0.8)

    print("trainLength : ",trainLength)

    traindata = data[:trainLength, :Xcolumns]
    trainlabel = data[:trainLength, Xcolumns:Xcolumns+Ycolumns]
    testdata = data[trainLength:, :Xcolumns]
    testlabel = data[trainLength:, Xcolumns:Xcolumns+Ycolumns]

    # print("traindata.shape : ",traindata.shape)
    # print("trainlabel.shape : ", trainlabel.shape)
    # print("testdata.shape : ", testdata.shape)
    # print("testlabel.shape : ", testlabel.shape)

    pd.DataFrame(testlabel.reshape(-1,1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\Sunsplot\\test\\testReal.csv",index=False)
    pd.DataFrame(trainlabel.reshape(-1,1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\Sunsplot\\train\\trainReal.csv", index=False)

    return traindata, trainlabel, testdata, testlabel


class ESN(object):

    def __init__(self, resSize, rho=0.9, cr=0.05, leaking_rate=0.2, W=None):
        """
        结果是产生储备池状态矩阵W
        :param resSize: reservoir size 储备池神经元个数
        :param rho: spectral radius 谱半径
        :param cr: connectivity ratio 连接率也可以叫稀疏度
        :param leaking_rate: leaking rate 泄露率
        :param W: predefined ESN reservoir 储备池连接权重
        """
        self.resSize = resSize
        self.leaking_rate = leaking_rate

        if W is None:
            # generate the ESN reservoir
            N = resSize * resSize  # 是一个方阵
            W = np.random.rand(N) - 0.5  # 随机生成一个 N*N的随机矩阵，每个元素取值在 [-0.5,0.5)
            zero_index = np.random.permutation(N)[int(N * cr * 1.0):]  # 随机生成需要被设置为0的元素索引
            W[zero_index] = 0  # [int(N * cr * 1.0):]这么多个数需要设置为0
            W = W.reshape((self.resSize, self.resSize))  # 变身成为N*N的矩阵
            # Option 1 - direct scaling (quick&dirty, reservoir-specific):
            # self.W *= 0.135
            # Option 2 - normalizing and setting spectral radius (correct, slow):
            print('ESN init: Setting spectral radius...ESN的构造函数进行中')
            rhoW = max(abs(linalg.eig(W)[0]))  # 求出W的最大特征值，也就是谱半径
            print('done.')
            W *= rho / rhoW  # 乘以一个小于1的因子
        else:
            assert W.shape[0] == W.shape[1] == resSize, "reservoir size mismatch，储备池不是方阵"
        self.W = W

    def __init_states__(self, X, initLen=0, reset_state=True):
        # allocate memory for the collected states matrix，S用来收集状态，形状是（样本数，inSize+resSize）,激活函数里是这三项的和
        self.S = np.zeros((len(X) - initLen, self.inSize + self.resSize))
        print("initial S : ，在__init_states__函数中初始化储备池状态", self.S.shape)
        if reset_state:
            self.s = np.zeros(self.resSize)
        s = self.s.copy()

        # run the reservoir with the data and collect S
        for t, u in enumerate(X):  # enumerate(X)：把X数组组合成（下标，数据）的形式，t是下标，u是
            # 状态更新公式，np.hstack：纵轴方向上堆加；np.hstack((1, u))在u的左边堆一个1
            s = (1 - self.leaking_rate) * s + self.leaking_rate * np.tanh(
                np.dot(self.Win, np.hstack(u)) + np.dot(self.W, s))
            if t >= initLen:
                self.S[t - initLen] = np.hstack((u, s))
        print("S的形状是：", self.S.shape)  # (13850, 606)
        if reset_state:
            self.s = s

    def setWinSize(self, X, init_states=True):
        self.inSize = 1 if np.ndim(X) == 1 else X.shape[1]
        if init_states:
            print("ESN fit_ridge: Initializing states...")
            self.Win = (np.random.rand(self.resSize, self.inSize) - 0.5) * 1
            self.__init_states__(X, initLen)
            print("done.")

    def fit(self, X, y, lmbd=1e-6, initLen=100, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t,) or (t, d), where
        :         t - length of time series, d - dimensionality.
        :param y : array-like, shape (t,). Target vector relative to X.
        :param lmbd: regularization lambda 正规化λ
        :param initLen: Number of samples to wash out the initial random state
        :param init_states: False allows skipping states initialization if
        :                   it was initialized before (with same X).
        :                   Useful in experiments with different targets.
        """
        # assert 表达式，字符串：表达式为False的时候，打印字符串，报出异常，程序终止
        assert len(X) == len(y), "input lengths mismatch."
        self.inSize = 1 if np.ndim(X) == 1 else X.shape[1]
        if init_states:
            print("ESN fit_ridge: Initializing states...在fit函数中初始化Win和S")
            self.Win = (np.random.rand(self.resSize, self.inSize) - 0.5) * 1
            self.__init_states__(X, initLen)
            print("done.")
        # alpha越大表示正则化强度越大，fit_intercept表示是否要将偏差添加到决策函数中，solver解算器的种类是奇异值分解方法，tol计算精度的设置
        # 通过奇异值分解能够计算出一个输出权值矩阵Wout
        self.ridge = Ridge(alpha=lmbd, fit_intercept=False,
                           solver='svd', tol=1e-6)
        # 通过岭回归的方法来训练储备池状态矩阵
        self.ridge.fit(self.S, y[initLen:])
        # print("模型权值：", self.ridge.coef_)
        print("模型权值形状：", np.array(self.ridge.coef_).shape)  # (1, 606)
        # print("模型偏置：", self.ridge.intercept_)
        return self

    def predict(self, X, init_states=True):
        """
        :param X: 1- or 2-dimensional array-like, shape (t) or (t, d)
        :param init_states: see above
        """
        if init_states:
            # assume states initialized with training data and we continue from there.
            self.__init_states__(X, 0, reset_state=False)
        print("S : {}".format(self.S.shape))  # (150, 606)
        y = self.ridge.predict(self.S)  # 用岭回归的方法来实现训练预测
        print("y.shape : {}".format(y.shape))  # (150, 1)
        return y


def getMAE(predict , real):
    return mean_absolute_error(real, predict)

def getSMAPE(predict , real):
    return np.mean(np.abs(predict - real) / (np.abs(predict) + np.abs(real)))

def getMSE(predict , real):
    return mean_squared_error(real, predict)

def getRMSE(predict , real):
    MSE = getMSE(predict, real)
    return np.sqrt(MSE)

def getR2(predict , real):
    # average = np.sum(real) / len(real)
    # return 1 - (np.sum(np.dot((real - predict).T, (real - predict))) / np.sum(np.dot((real - average).T, (real - average))))
    return r2_score(real, predict)

def getR(predict ,real):
    '''
    以下是绝对值的区间范围，R本身取值在[-1，1]之间
    0.8-1.0     极强相关
    0.6-0.8     强相关
    0.4-0.6     中等程度相关
    0.2-0.4     弱相关
    0.0-0.2     极弱相关或无相关
    '''
    predict = np.squeeze(predict) #去掉多余的维度
    real = np.squeeze(real)
    return pearsonr(real, predict)[0]


if __name__ == '__main__':
    '''
     #房山数据集的main程序

    for resSizeNumber in range(400, 1000, 10):
        
        path = 'E:\\yan_1\\BLS_self\\fangshan.csv'
        traindata, trainlabel, testdata, testlabel = getDataFromCSV(path)
        initLength = 100

        resSize = resSizeNumber
        rho = 0.95  # spectral radius
        cr = 0.05  # connectivity ratio
        leaking_rate = 0.1  # leaking rate
        lmbd = 0.001  # regularization coefficient

        esn = ESN(resSize=resSize, rho=rho, cr=cr, leaking_rate=leaking_rate)

        # scaler1 = MinMaxScaler()
        # scaler2 = MinMaxScaler()
        # scaler3 = MinMaxScaler()
        # scaler4 = MinMaxScaler()
        # data = scaler1.fit_transform(data)

        x_train = traindata
        y_trian = trainlabel
        Xtest =testdata
        ytest = testlabel

        print(x_train.shape, y_trian.shape, Xtest.shape, ytest.shape)
        # print("len(x_train) : {}, len(y_trian) : {}".format(len(x_train),len(y_trian)))
        print('*' * 100)
        starttime = datetime.datetime.now()
        esn.fit(x_train, y_trian, initLen=0, lmbd=lmbd)
        endtime = datetime.datetime.now()
        print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

        starttime1 = datetime.datetime.now()
        y_predicted = esn.predict(Xtest)
        endtime1 = datetime.datetime.now()
        # print("y_predicted shape : ",y_predicted.shape)
        y_predicted = y_predicted
        ytest = ytest

        # 计算指标
        # mse = mean_squared_error(ytest, y_predicted)
        MAE = getMAE(y_predicted[initLength:], ytest[initLength:])
        SMAPE = getSMAPE(y_predicted[initLength:], ytest[initLength:])
        RMSE = getRMSE(y_predicted[initLength:], ytest[initLength:])
        R2 = getR2(y_predicted[initLength:], ytest[initLength:])
        # R = getR(y_predicted[initLength:], ytest[initLength:])

        RMSEList.append(RMSE)
        MAEList.append(MAE)
        SMAPEList.append(SMAPE)
        R2List.append(R2)
        # RList.append(R)
        resSizeNumberList.append(resSizeNumber)
        trainTimeList.append((endtime - starttime).total_seconds())
        testTimeList.append((endtime1 - starttime1).total_seconds())

        print('MAE : ', MAE)
        print('SMAPE : ', SMAPE)
        print('RMSE : ', RMSE)
        print('R2 : ', R2)
        predictlabel = np.array(y_predicted).T.reshape(-1, 1)
        pd.DataFrame(predictlabel).to_csv(
            "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\ESN\\test\\ESN_resSize_{}.csv".format(resSizeNumber),
                index=False)
        print("predictlabel.shape:", predictlabel.shape)
    RMSEList = np.array(RMSEList)
    MAEList = np.array(MAEList)
    SMAPEList = np.array(SMAPEList)
    R2List = np.array(R2List)
    RList = np.array(RList)
    resSizeNumberList = np.array(resSizeNumberList)
    trainTimeList = np.array(trainTimeList)
    testTimeList = np.array(testTimeList)

    datafram = {'resSizeNumberList': resSizeNumberList.reshape(-1, ),
                'RMSE': RMSEList.reshape(-1, ), 'MAE': MAEList.reshape(-1, ),
                'SMAPE': SMAPEList.reshape(-1, ), 'R2': R2List.reshape(-1, ),
                'trainTime': trainTimeList.reshape(-1, ), 'testTime': testTimeList.reshape(-1, ), }
    pd.DataFrame(datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\ESN_huizong_test.csv', index=False)
        
                
        '''
    '''
    #MSO_8数据集

    X_sample_size = 12
    Y__sample_size = 1
    slide_size = 1

    generate_data_by_slide(X_sample_size, Y__sample_size, slide_size)

    path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\MSO_8_slide{}_data.csv".format(slide_size)
    traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
    initLength = 100
    for resSizeNumber in range(400, 1000, 10):

        traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
        initLength = 100

        resSize = resSizeNumber
        rho = 0.95  # spectral radius
        cr = 0.05  # connectivity ratio
        leaking_rate = 0.1  # leaking rate
        lmbd = 0.001  # regularization coefficient

        esn = ESN(resSize=resSize, rho=rho, cr=cr, leaking_rate=leaking_rate)

        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        # data = scaler1.fit_transform(data)

        x_train = scaler1.fit_transform(traindata)
        y_trian = scaler2.fit_transform(trainlabel)
        Xtest = scaler1.fit_transform(testdata)
        ytest = scaler2.fit_transform(testlabel)

        print(x_train.shape, y_trian.shape, Xtest.shape, ytest.shape)
        # print("len(x_train) : {}, len(y_trian) : {}".format(len(x_train),len(y_trian)))
        print('*' * 100)
        starttime = datetime.datetime.now()
        esn.fit(x_train, y_trian, initLen=0, lmbd=lmbd)
        endtime = datetime.datetime.now()
        print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

        starttime1 = datetime.datetime.now()
        y_predicted = esn.predict(Xtest)
        endtime1 = datetime.datetime.now()
        # print("y_predicted shape : ",y_predicted.shape)
        y_predicted = scaler2.inverse_transform(y_predicted).reshape(-1, 1)
        ytest = scaler2.inverse_transform(ytest).reshape(-1, 1)

        # 计算指标
        # mse = mean_squared_error(ytest, y_predicted)
        MAE = getMAE(y_predicted[initLength:], ytest[initLength:])
        SMAPE = getSMAPE(y_predicted[initLength:], ytest[initLength:])
        RMSE = getRMSE(y_predicted[initLength:], ytest[initLength:])
        R2 = getR2(y_predicted[initLength:], ytest[initLength:])
        # R = getR(y_predicted[initLength:], ytest[initLength:])

        RMSEList.append(RMSE)
        MAEList.append(MAE)
        SMAPEList.append(SMAPE)
        R2List.append(R2)
        # RList.append(R)
        resSizeNumberList.append(resSizeNumber)
        trainTimeList.append((endtime - starttime).total_seconds())
        testTimeList.append((endtime1 - starttime1).total_seconds())

        print('MAE : ', MAE)
        print('SMAPE : ', SMAPE)
        print('RMSE : ', RMSE)
        print('R2 : ', R2)
        predictlabel = np.array(y_predicted).T.reshape(-1, 1)
        pd.DataFrame(predictlabel).to_csv(
            "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\ESN\\test\\ESN_resSize_{}.csv".format(resSizeNumber),
            index=False)
        print("predictlabel.shape:", predictlabel.shape)
    RMSEList = np.array(RMSEList)
    MAEList = np.array(MAEList)
    SMAPEList = np.array(SMAPEList)
    R2List = np.array(R2List)
    RList = np.array(RList)
    resSizeNumberList = np.array(resSizeNumberList)
    trainTimeList = np.array(trainTimeList)
    testTimeList = np.array(testTimeList)

    datafram = {'resSizeNumberList': resSizeNumberList.reshape(-1, ),
                'RMSE': RMSEList.reshape(-1, ), 'MAE': MAEList.reshape(-1, ),
                'SMAPE': SMAPEList.reshape(-1, ), 'R2': R2List.reshape(-1, ),
                'trainTime': trainTimeList.reshape(-1, ), 'testTime': testTimeList.reshape(-1, ), }
    pd.DataFrame(datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\MSO_8\\ESN_huizong_test.csv', index=False)

    '''
    #Sunsplot数据集
    X_sample_size = 3
    Y__sample_size = 1
    slide_size = 1

    generate_data_by_slide(X_sample_size, Y__sample_size, slide_size)

    path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\Sunsplot\\Sunsplot_slide{}_data.csv".format(slide_size)
    traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
    initLength = 100
    for resSizeNumber in range(400, 1000, 10):

        traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
        initLength = 100

        resSize = resSizeNumber
        rho = 0.95  # spectral radius
        cr = 0.05  # connectivity ratio
        leaking_rate = 0.1  # leaking rate
        lmbd = 0.001  # regularization coefficient

        esn = ESN(resSize=resSize, rho=rho, cr=cr, leaking_rate=leaking_rate)

        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        # data = scaler1.fit_transform(data)

        x_train = scaler1.fit_transform(traindata)
        y_trian = scaler2.fit_transform(trainlabel)
        Xtest = scaler1.fit_transform(testdata)
        ytest = scaler2.fit_transform(testlabel)

        print(x_train.shape, y_trian.shape, Xtest.shape, ytest.shape)
        # print("len(x_train) : {}, len(y_trian) : {}".format(len(x_train),len(y_trian)))
        print('*' * 100)
        starttime = datetime.datetime.now()
        esn.fit(x_train, y_trian, initLen=0, lmbd=lmbd)
        endtime = datetime.datetime.now()
        print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

        starttime1 = datetime.datetime.now()
        y_predicted = esn.predict(Xtest)
        endtime1 = datetime.datetime.now()
        # print("y_predicted shape : ",y_predicted.shape)
        y_predicted = scaler2.inverse_transform(y_predicted).reshape(-1, 1)
        ytest = scaler2.inverse_transform(ytest).reshape(-1, 1)

        # 计算指标
        # mse = mean_squared_error(ytest, y_predicted)
        MAE = getMAE(y_predicted[initLength:], ytest[initLength:])
        SMAPE = getSMAPE(y_predicted[initLength:], ytest[initLength:])
        RMSE = getRMSE(y_predicted[initLength:], ytest[initLength:])
        R2 = getR2(y_predicted[initLength:], ytest[initLength:])
        # R = getR(y_predicted[initLength:], ytest[initLength:])

        RMSEList.append(RMSE)
        MAEList.append(MAE)
        SMAPEList.append(SMAPE)
        R2List.append(R2)
        # RList.append(R)
        resSizeNumberList.append(resSizeNumber)
        trainTimeList.append((endtime - starttime).total_seconds())
        testTimeList.append((endtime1 - starttime1).total_seconds())

        print('MAE : ', MAE)
        print('SMAPE : ', SMAPE)
        print('RMSE : ', RMSE)
        print('R2 : ', R2)
        predictlabel = np.array(y_predicted).T.reshape(-1, 1)
        pd.DataFrame(predictlabel).to_csv(
            "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\Sunsplot\\ESN\\test\\ESN_resSize_{}.csv".format(resSizeNumber),
            index=False)
        print("predictlabel.shape:", predictlabel.shape)
    RMSEList = np.array(RMSEList)
    MAEList = np.array(MAEList)
    SMAPEList = np.array(SMAPEList)
    R2List = np.array(R2List)
    RList = np.array(RList)
    resSizeNumberList = np.array(resSizeNumberList)
    trainTimeList = np.array(trainTimeList)
    testTimeList = np.array(testTimeList)

    datafram = {'resSizeNumberList': resSizeNumberList.reshape(-1, ),
                'RMSE': RMSEList.reshape(-1, ), 'MAE': MAEList.reshape(-1, ),
                'SMAPE': SMAPEList.reshape(-1, ), 'R2': R2List.reshape(-1, ),
                'trainTime': trainTimeList.reshape(-1, ), 'testTime': testTimeList.reshape(-1, ), }
    pd.DataFrame(datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\Sunsplot\\ESN_huizong_test.csv', index=False)

