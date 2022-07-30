import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time
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

def LCFBLS(train_x, train_y, test_x, test_y, s, c, N1, N2, N3):
    # 预处理数据，axis等于1时标准化每个样本（行）,axis等于0时独立地标准化每个特征
    # train_x = preprocessing.scale(train_x, axis=1)
    # 将输入矩阵进行行链接，即平铺展开整个矩阵,np.hstack：纵轴方向上堆加；np.hstack((1, u))在u的左边堆一个1
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])  # 12000 * 7
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0], N2 * N1])  # 先规范了映射层输出矩阵的大小
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    train_time = 0
    test_time = 0

    #输入数据到第一个映射层窗口的特殊处理
    random.seed(0)
    weightOfEachWindow = 2 * random.randn(train_x.shape[1] + 1, N1) - 1 #7*N1
    FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow) #12000*N1

    scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 0.05)).fit(FeatureOfEachWindow)#12000*N1
    FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)#12000*N1
    betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T #(12000*N1, 12000*7)=>7*N1
    Beta1OfEachWindow.append(betaOfEachWindow)

    outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow) #12000*N1
    distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
    minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
    outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[0]) / distOfMaxAndMin[0]
    # 生成映射节点最终输入
    OutputOfFeatureMappingLayer[:, 0:N1] = outputOfEachWindow


    time_start = datetime.datetime.now()# 计时开始
    for i in range(N2-1):  # N2 是 映射层窗口数
        random.seed(i+1)
        # 随机化权重，N1是映射层每个窗口内节点数
        weightOfEachWindow = 2 * random.randn(N1, N1) - 1

        # 源输入X的平铺展开矩阵与随机化权重W相乘 = 每个窗口的特征结点，FeatureOfEachWindow是映射层的特征节点
        FeatureOfInputDataWithBias = OutputOfFeatureMappingLayer[:, i*N1:(i+1)*N1] # 12000*N1
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow) # 12000 * N1

        # 对上述结果归一化处理，改动feature_range的范围值会使得网络更好,fangshan=0.05,daxing=0.001
        scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 0.05)).fit(FeatureOfEachWindow)
        # 进行标准化
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(FeatureOfEachWindow)

        # 随机化贝塔值，betaOfEachWindow返回的是每个映射节点的贝塔值
        betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
        # print("betaOfEachWindow.shape : {}".format(betaOfEachWindow.shape)) #
        Beta1OfEachWindow.append(betaOfEachWindow)

        # 源输入X的平铺展开矩阵与贝塔值相乘 = 每个窗口的输出
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
        #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))

        # 求解输出最大值与最小值的距离
        distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
        outputOfEachWindow = (outputOfEachWindow - minOfEachWindow[i+1]) / distOfMaxAndMin[i+1]

        # 生成映射节点最终输入
        OutputOfFeatureMappingLayer[:, N1 * (i+1):N1 * (i + 2)] = outputOfEachWindow
        # del outputOfEachWindow
        # del FeatureOfEachWindow
        # del weightOfEachWindow

    # 生成增强结点,LCFBLS在CFBLS的基础上改变的是强化层输入，这个输入是映射层最后一个窗口的输出
    # 1. InputOfEnhanceLayerWithBias返回的是映射层全部输出+一列偏置的组合矩阵,LCFBLS的结构是强化层的输入时映射层最后一个窗口的输出
    InputOfEnhanceLayerWithBias = np.hstack(
        [OutputOfFeatureMappingLayer[:,(N2-1)*N1:N1*N2], 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))]) #12000*N1

    # N3 => 强化层节点数，LA.orth（A）是使用SVD奇异值分解法构造A的正交基矩阵
    if N1 * N2 >= N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N1 + 1, N3).T - 1).T

    # 增强结点乘上相应随机权重
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, weightOfEnhanceLayer) #12000*N3
    # print("tempOfOutputOfEnhanceLayer.type : ", type(tempOfOutputOfEnhanceLayer))
    # print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    # 参数的归一化？ s是收敛系数
    parameterOfShrink = s / np.max(tempOfOutputOfEnhanceLayer)
    # 增强节点的最终输出,强化层通过非线性映射激活函数tansig，作为最后的输出
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink) #12000*N3

    # 最终输入 = 映射结点最终输入 + 增强结点最终输入 的平铺展开，组合矩阵A
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])
    # 求解组合矩阵的伪逆，c是正则化系数
    pinvOfInput = pinv(InputOfOutputLayer, c)
    # 输出权重 = 组合矩阵的伪逆 * 输出Y
    OutputWeight = np.dot(pinvOfInput, train_y)

    # 计算训练时间
    time_end =  datetime.datetime.now()  # 计时结束
    train_time = (time_end - time_start).total_seconds()

    # 训练预测输出 = 最终输入 乘以 输出权重
    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)

    # 测试过程

    # 标准化处理测试集输入
    # test_x = preprocessing.scale(test_x, axis=1)
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], N2 * N1])
    time_start = datetime.datetime.now()

    #测试集输入第一个映射窗口，这里单独做特殊处理
    outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, Beta1OfEachWindow[0])
    OutputOfFeatureMappingLayerTest[:,0:N1] = (ymax - ymin) * (
            outputOfEachWindowTest - minOfEachWindow[0]) / distOfMaxAndMin[0] - ymin

    # 测试集上的映射层的输入权值矩阵应该和训练集上的输入权值矩阵是一样的，所以使用Beta1OfEachWindow来搜集，之后的映射窗口输入是前一个映射窗口输出
    #CFBLS模型中需要注意的是测试集验证中，OutputOfFeatureMappingLayerTest与Beta1OfEachWindow的对应关系
    for i in range(N2-1):
        outputOfEachWindowTest = np.dot(OutputOfFeatureMappingLayerTest[:,N1 * i:N1 * (i + 1)], Beta1OfEachWindow[i+1])
        OutputOfFeatureMappingLayerTest[:, N1 * (i+1):N1 * (i + 2)] = (ymax - ymin) * (
                    outputOfEachWindowTest - minOfEachWindow[i+1]) / distOfMaxAndMin[i+1] - ymin

    #测试集强化层输入需要做特殊处理，就是将映射层最后一个窗口的输出作为强化层的输入
    InputOfEnhanceLayerWithBiasTest = np.hstack(
        [OutputOfFeatureMappingLayerTest[:, (N2-1)*N1:N1*N2], 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, weightOfEnhanceLayer)

    # parameterOfShrink也是和之前训练集上的parameterOfShrink是一样的
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)

    # 测试集最终输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
    # print("LCFBLS : ",InputOfOutputLayerTest.shape)

    # 测试集预测输出 = 测试集最终输入 乘以 输出权重
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    time_end = datetime.datetime.now()
    test_time = (time_end - time_start).total_seconds()

    # testAcc = show_accuracy(OutputOfTest, test_y)
    # print('Testing accurate is', testAcc * 100, '%')
    # print('Testing time is ', testTime, 's')
    # test_acc[0][0] = testAcc
    # test_time[0][0] = testTime

    return OutputOfTrain, OutputOfTest, train_time, test_time

