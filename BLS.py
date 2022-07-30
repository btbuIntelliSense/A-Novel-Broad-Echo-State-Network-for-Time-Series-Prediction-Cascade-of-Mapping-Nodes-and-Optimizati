# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:09:38 2018

@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python.
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper,
   please contact the authors of related paper.
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import datetime



def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count / len(Label), 5))

def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    # 通过最小化二范式目标函数，推导出的A的伪逆
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


'''
参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长
'''


def BLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):
    L = 0
    # 预处理数据，axis等于1时标准化每个样本（行）,axis等于0时独立地标准化每个特征
    # train_x = preprocessing.scale(train_x, axis=1)
    # 将输入矩阵进行行链接，即平铺展开整个矩阵,np.hstack：纵轴方向上堆加；np.hstack((1, u))在u的左边堆一个1
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])  # 偏置是0.1
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])  # 先规范了映射层输出矩阵的大小
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    train_acc_all = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = 0
    test_time = 0
    time_start = datetime.datetime.now()  # 计时开始
    for i in range(N2):  # N2 是 映射层窗口数
        random.seed(i)
        # 随机化权重，N1是映射层每个窗口内节点数
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1

        # 源输入X的平铺展开矩阵与随机化权重W相乘 = 每个窗口的特征结点，FeatureOfEachWindow是映射层的特征节点
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)

        # 对上述结果归一化处理,fangshan=0.05,daxing=0.001
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 0.05)).fit(FeatureOfEachWindow)
        # 进行标准化
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)

        # 随机化贝塔值，betaOfEachWindow返回的是每个映射节点的贝塔值
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        # print("betaOfEachWindow.shape : {}".format(betaOfEachWindow.shape))
        Beta1OfEachWindow.append(betaOfEachWindow)

        # 源输入X的平铺展开矩阵与贝塔值相乘 = 每个窗口的输出
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        # print("outputOfEachWindow.shape : {}".format(outputOfEachWindow.shape))
        # 求解输出最大值与最小值的距离
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))

        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]

        # 生成映射节点最终输入
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        # del outputOfEachWindow
        # del FeatureOfEachWindow
        # del weightOfEachWindow

    # 生成增强结点
    # 1. InputOfEnhanceLayerWithBias返回的是映射层全部输出+一列偏置的组合矩阵
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

    # N3 => 强化层节点数，LA.orth（A）是使用SVD奇异值分解法构造A的正交基矩阵
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    # 增强结点乘上相应随机权重
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    # print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    # 参数的归一化？ s是收敛系数
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    # 增强节点的最终输出,强化层通过非线性映射激活函数tansig，作为最后的输出
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 最终输入 = 映射结点最终输入 + 增强结点最终输入 的平铺展开，组合矩阵A
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    # 求解组合矩阵的伪逆，c是正则化系数
    pinvOfInput = pinv(InputOfOutputLayer, c)
    # 输出权重 = 组合矩阵的伪逆 * 输出Y
    OutputWeight = np.dot(pinvOfInput, train_y)

    # 计算训练时间
    time_end = datetime.datetime.now()  # 计时结束
    train_time = (time_end - time_start).total_seconds()

    # 训练预测输出 = 最终输入 乘以 输出权重
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)

    # trainAcc = show_accuracy(OutputOfTrain, train_y)
    # print('Training accurate is', trainAcc * 100, '%')
    # print('Training time is ', trainTime, 's')
    # train_acc_all[0][0] = trainAcc
    # train_time[0][0] = trainTime

    # 测试过程

    # 标准化处理测试集输入
    # test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = datetime.datetime.now()  # 计时开始

    # 测试集上的映射层的输入权值矩阵应该和训练集上的输入权值矩阵是一样的，所以使用Beta1OfEachWindow来搜集，
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i] - ymin

    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    # parameterOfShrink也是和之前训练集上的parameterOfShrink是一样的
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    # 测试集最终输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

    # 测试集预测输出 = 测试集最终输入 乘以 输出权重
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = datetime.datetime.now()  # 计时结束
    test_time = (time_end - time_start).total_seconds()

    # testAcc = show_accuracy(OutputOfTest, test_y)
    # print('Testing accurate is', testAcc * 100, '%')
    # print('Testing time is ', testTime, 's')
    # test_acc[0][0] = testAcc
    # test_time[0][0] = testTime

    return OutputOfTrain, OutputOfTest, train_time, test_time


'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长
'''


def BLS_AddEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M):
    # 生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    # train_x = preprocessing.scale(train_x, axis=1)  # 处理数据
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])

    distOfMaxAndMin = []
    minOfEachWindow = []

    train_acc = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])

    time_start = time.time()  # 计时开始
    Beta1OfEachWindow = []

    # N2 是 映射层窗口数，N1是每个窗口内映射节点的数量
    for i in range(N2):
        random.seed(i + u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        # FeatureOfEachWindow是未归一化的映射层输出
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)

        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        # FeatureOfEachWindowAfterPreprocess是归一化后的映射层输出
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)

        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)

        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    # 强化层的经过非线性映射后的输出
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])

    # 求出的是组合矩阵的伪逆
    pinvOfInput = pinv(InputOfOutputLayer, c)

    # 组合矩阵的伪逆乘以标签等于输出权值矩阵
    OutputWeight = pinvOfInput.dot(train_y)

    time_end = time.time()
    trainTime = time_end - time_start

    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    # trainAcc = show_accuracy(OutputOfTrain, train_y)
    # print('Training accurate is', trainAcc * 100, '%')
    # print('Training time is ', trainTime, 's')
    # train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (outputOfEachWindowTest - minOfEachWindow[i]) / distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()  # 训练完成
    testTime = time_end - time_start
    # testAcc = show_accuracy(OutputOfTest, test_y)
    # print('Testing accurate is', testAcc * 100, '%')
    # print('Testing time is ', testTime, 's')
    # test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    '''
        增量增加强化节点
        L------步数
        M------步长
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start = time.time()
        if N1 * N2 >= M:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M) - 1)
        else:
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2 * N1 + 1, M).T - 1).T

        # InputOfEnhanceLayerWithBias还是刚开始映射层的输出
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s / np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd * parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer, OutputOfEnhanceLayerAdd])

        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T, D)).I.dot(np.dot(D.T, pinvOfInput))  # 这个地方公式错了吧
        else:
            B = pinv(C, c)
        # 增加一个强化层节点后的新的组合矩阵的伪逆
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])

        # 增加一个强化层节点后的输出权值矩阵，
        OutputWeightEnd = pinvOfInput.dot(train_y)

        # 增加一个强化层节点后的新的组合矩阵
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e + 1] = Training_time

        # 增加一个强化层节点后的新的输出预测序列
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        # print("OutputOfTrain1.shape {}".format(OutputOfTrain1.shape))
        # TrainingAccuracy = show_accuracy(OutputOfTrain1, train_y)
        # train_acc[0][e + 1] = TrainingAccuracy
        # print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %')

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(
            InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        # 增加一个强化层节点后的新的测试预测序列输出
        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest1, test_y)
        # print("OutputOfTest1.shape {}".format(OutputOfTest1.shape))

        Test_time = time.time() - time_start
        test_time[0][e + 1] = Test_time
        # test_acc[0][e + 1] = TestingAcc
        # print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %')

    return OutputOfTrain1, OutputOfTest1 # 因为是增加强化层节点，这里应该返回的是 OutputOfTrain1 和 OutputOfTest1


'''
增加映射层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
L------步数

M1-----增加映射节点数
M2-----与增加映射节点对应的强化节点数
M3-----新增加的强化节点
'''


def BLS_AddFeatureEnhanceNodes(train_x, train_y, test_x, test_y, s, c, N1, N2, N3, L, M1, M2, M3):
    # 生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0

    train_x = preprocessing.scale(train_x, axis=1)
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])

    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1, L + 1])
    test_acc = np.zeros([1, L + 1])
    train_time = np.zeros([1, L + 1])
    test_time = np.zeros([1, L + 1])
    time_start = time.time()  # 计时开始
    '''
    N1-----映射层每个窗口内节点数
    N2-----映射层窗口数
    N3-----强化层节点数
    '''
    for i in range(N2):
        random.seed(i + u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)

        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)

        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)

        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.mean(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i]) / distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:, N1 * i:N1 * (i + 1)] = outputOfEachWindow
        del outputOfEachWindow
        del FeatureOfEachWindow
        del weightOfEachWindow

        # 生成强化层

    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3) - 1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T

    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer)
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)

    # 生成组合矩阵InputOfOutputLayerTrain
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain, c)
    OutputWeight = pinvOfInput.dot(train_y)  # 全局违逆
    time_end = time.time()  # 训练完成
    trainTime = time_end - time_start

    OutputOfTrain = np.dot(InputOfOutputLayerTrain, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime

    test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = time.time()

    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:, N1 * i:N1 * (i + 1)] = (outputOfEachWindowTest - minOfEachWindow[i]) / \
                                                                  distOfMaxAndMin[i]

    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])

    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = time.time()
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest, test_y)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增加 映射节点 和 强化节点
        L------新增加的映射层窗口数
        M1-----每次新增加映射节点数，也就是每一个窗口的映射单元数量
        M2-----与增加映射节点对应的强化节点数
        M3-----新增加的强化节点
    '''
    WeightOfNewFeature2 = list()  # WeightOfNewFeature2搜集的是新增加映射节点连接对应强化节点的权重
    WeightOfNewFeature3 = list()  # WeightOfNewFeature3搜集的是强化层内部的连接权重
    #
    for e in list(range(L)):
        time_start = time.time()
        random.seed(e + N2 + u)
        # 每次新增加映射层节点数量为M1
        weightOfNewMapping = 2 * random.random([train_x.shape[1] + 1, M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)

        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)

        betaOfNewWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        # Beta1OfEachWindow里边还包括未加入映射节点之前的β
        Beta1OfEachWindow.append(betaOfNewWindow)

        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append(np.max(TempOfFeatureOutput, axis=0) - np.min(TempOfFeatureOutput, axis=0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput, axis=0))
        # 注意这里是N2+e，OutputOfFeatureMappingLayer是新增加映射节点后的映射层输出，NewInputOfEnhanceLayerWithBias是加了偏置后的映射层输出
        outputOfNewWindow = (TempOfFeatureOutput - minOfEachWindow[N2 + e]) / distOfMaxAndMin[N2 + e]
        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer, outputOfNewWindow])
        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0], 1))])

        '''
        M1-----每次新增加映射节点数，也就是每一个窗口的映射单元数量
        M2-----与增加映射节点对应的强化节点数
        '''
        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2 * random.random([M1 + 1, M2]) - 1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2 * random.random([M1 + 1, M2]).T - 1).T

        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)

        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)

        parameter1 = s / np.max(tempOfNewFeatureEhanceNodes)

        # outputOfNewFeatureEhanceNodes表示新增加映射节点的输出
        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        if N2 * N1 + e * M1 >= M3:
            random.seed(67797325 + e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2 * N1 + (e + 1) * M1 + 1, M3) - 1)
        else:
            random.seed(67797325 + e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2 * N1 + (e + 1) * M1 + 1, M3).T - 1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)

        # InputOfEnhanceLayerWithBias表示强化层的输入
        InputOfEnhanceLayerWithBias = np.hstack(
            [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s / np.max(tempOfNewEnhanceNodes)

        # OutputOfNewEnhanceNodes表示的是新增加强化层节点的输出
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2)

        # OutputOfTotalNewAddNodes表示新的组合矩阵
        OutputOfTotalNewAddNodes = np.hstack(
            [outputOfNewWindow, outputOfNewFeatureEhanceNodes, OutputOfNewEnhanceNodes])

        # InputOfOutputLayerTrain这个需要注意一下，表示的是没有新增加任何结点之前的组合矩阵
        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain, OutputOfTotalNewAddNodes])
        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)

        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w) - D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C, c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)), B])
        OutputWeight = pinvOfInput.dot(train_y)
        InputOfOutputLayerTrain = tempOfInputOfLastLayes

        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e + 1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        TrainingAccuracy = show_accuracy(predictLabel, train_y)
        train_acc[0][e + 1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %')

        # 测试过程
        # 先生成新映射窗口输出
        time_start = time.time()
        WeightOfNewMapping = Beta1OfEachWindow[N2 + e]

        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping)

        outputOfNewWindowTest = (outputOfNewWindowTest - minOfEachWindow[N2 + e]) / distOfMaxAndMin[N2 + e]

        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, outputOfNewWindowTest])

        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones([OutputOfFeatureMappingLayerTest.shape[0], 1])])

        NewInputOfEnhanceLayerWithBiasTest = np.hstack(
            [outputOfNewWindowTest, 0.1 * np.ones([outputOfNewWindowTest.shape[0], 1])])

        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]

        OutputOfRelateEnhanceNodes = tansig(
            NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)

        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]

        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes) * parameter2)

        InputOfOutputLayerTest = np.hstack(
            [InputOfOutputLayerTest, outputOfNewWindowTest, OutputOfRelateEnhanceNodes, OutputOfNewEnhanceNodes])

        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)

        TestingAccuracy = show_accuracy(predictLabel, test_y)
        time_end = time.time()
        Testing_time = time_end - time_start
        test_time[0][e + 1] = Testing_time
        test_acc[0][e + 1] = TestingAccuracy
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %')

    return OutputOfTrain, OutputOfTest




