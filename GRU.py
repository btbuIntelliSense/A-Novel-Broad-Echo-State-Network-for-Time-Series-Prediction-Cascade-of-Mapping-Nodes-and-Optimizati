# import pandas as pd
# import numpy as np
# from keras.models import Sequential
#
# import datetime
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from keras.layers import GRU, LSTM, Dropout, Dense
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# from sklearn.preprocessing import MinMaxScaler
#
# plt.rcParams['font.size'] = 16
# plt.rcParams['font.family'] = ['STKaiti']
# plt.rcParams['axes.unicode_minus'] = False
#
# name = 'Sunsplot'
#
# def data_process_slide(X, Y, X_sample_size, Y__sample_size, slide_size,iteration):
#     OUT = []
#     y = []
#     X.reshape(-1, 1)
#     Y.reshape(-1, 1)
#     for i in range(iteration):
#         OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
#         y.append(Y[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
#     OUT = np.array(OUT).reshape(iteration,X_sample_size)
#     y = np.array(y).reshape(iteration,Y__sample_size)
#     print("OUT.shape : ", OUT.shape)
#     print("y.shape : ", y.shape)
#     return OUT, y
#
# def generate_data_by_slide( X_sample_size, Y__sample_size, slide_size):
#     '''
#     前提是原始数据是一列数据，现在把它划分为（iteration,X_sample_size + Y__sample_size）
#     :param X_sample_size: X的样本个数
#     :param Y__sample_size: Y的样本个数
#     :param slide_size: 窗口的滑动步长
#     :return: None
#     '''
#     path = 'E:\yan_2\CFBLS_LCFBLS复现\dataset\Sunspot.csv'
#     X = pd.read_csv(path).values[1:2501, 2:3].reshape(2500, 1)
#
#     OUT = []
#     y = []
#
#     iteration = len(X) - X_sample_size - 1
#     for i in range(iteration):
#         OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
#         y.append(X[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
#     OUT = np.array(OUT).reshape(iteration,X_sample_size)
#     y = np.array(y).reshape(iteration,Y__sample_size)
#     print("OUT.shape : ", OUT.shape)
#     print("y.shape : ", y.shape)
#     data = np.hstack((OUT,y))
#
#     pd.DataFrame(data).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name,slide_size),
#                              index=False)
#
# def getDataFromCSV(path,Xcolumns,Ycolumns):
#     '''
#     将数据切片
#     :return: 没有归一化的数据
#     '''
#     data = pd.read_csv(path).values[1:2401, : ]
#     trainLength = int(data.shape[0] * 0.8)
#
#     print("trainLength : ",trainLength)
#
#     traindata = data[:trainLength, :Xcolumns]
#     trainlabel = data[:trainLength, Xcolumns:Xcolumns+Ycolumns]
#     testdata = data[trainLength:, :Xcolumns]
#     testlabel = data[trainLength:, Xcolumns:Xcolumns+Ycolumns]
#
#     print("traindata.shape : ",traindata.shape)
#     print("trainlabel.shape : ", trainlabel.shape)
#     print("testdata.shape : ", testdata.shape)
#     print("testlabel.shape : ", testlabel.shape)
#
#     # pd.DataFrame(testlabel.reshape(-1,1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\Sunsplot\\test\\testReal.csv",index=False)
#     # pd.DataFrame(trainlabel.reshape(-1,1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\Sunsplot\\train\\trainReal.csv", index=False)
#
#     return traindata, trainlabel, testdata, testlabel
#
#
#
# def getMAE(predict, real):
#     return mean_absolute_error(real, predict)
#
#
# def getSMAPE(predict, real):
#     return np.mean(np.abs(predict - real) / (np.abs(predict) + np.abs(real)))
#
#
# def getMSE(predict, real):
#     return mean_squared_error(real, predict)
#
#
# def getRMSE(predict, real):
#     MSE = getMSE(predict, real)
#     return np.sqrt(MSE)
#
#
# def getR2(predict, real):
#     # average = np.sum(real) / len(real)
#     # return 1 - (np.sum(np.dot((real - predict).T, (real - predict))) / np.sum(np.dot((real - average).T, (real - average))))
#     return r2_score(real, predict)
#
#
# def getR(predict, real):
#     '''
#     以下是绝对值的区间范围，R本身取值在[-1，1]之间
#     0.8-1.0     极强相关
#     0.6-0.8     强相关
#     0.4-0.6     中等程度相关
#     0.2-0.4     弱相关
#     0.0-0.2     极弱相关或无相关
#     '''
#     predict = np.squeeze(predict)  # 去掉多余的维度
#     real = np.squeeze(real)
#     return pearsonr(real, predict)[0]
#
# if __name__ == '__main__':
#
#     X_sample_size = 3
#     Y__sample_size = 1
#     slide_size = 1
#
#     generate_data_by_slide(X_sample_size, Y__sample_size, slide_size)
#
#     path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name,slide_size)
#     traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
#     initLength = 100
#
#     RMSE = []
#     MAE = []
#     SMAPE = []
#     R2 = []
#     R = []
#     trainTime = []
#     testTime = []
#
#     traindata = traindata.reshape(-1, 3, 1)
#     trainlabel = trainlabel.reshape(-1, 1)
#     testdata = testdata.reshape(-1, 3, 1)
#     testlabel = testlabel.reshape(-1, 1)
#
#     model = Sequential()
#     model.add(GRU(32, input_shape=(traindata.shape[1], traindata.shape[2]), return_sequences=True,
#                     name='LSTM_S0', kernel_initializer='orthogonal'))
#     model.add(GRU(64, return_sequences=False,
#                     name='LSTM_S1'))
#     model.add(Dense(1))
#     model.compile(loss='mae', optimizer='adam')
#     print(model.summary())
#     starttime = datetime.datetime.now()
#     model.fit(traindata, trainlabel, epochs=100, batch_size=32, verbose=2, shuffle=False)
#     endtime = datetime.datetime.now()
#     trainTime = (endtime - starttime).total_seconds()
#
#     starttime1 = datetime.datetime.now()
#     predictlabel = model.predict(testdata).reshape(-1, 1)
#     endtime1 = datetime.datetime.now()
#     testTime = (endtime1 - starttime1).total_seconds()
#
#     predictlabel = predictlabel.reshape(len(testdata), 1)
#     testlabel = testlabel.reshape(-1,1)
#
#     endtime = datetime.datetime.now()
#
#     pd.DataFrame(predictlabel.reshape(-1, 1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO\\GRU\\test\\GRU_MSO_predict.csv",index=False)
#
#     # 计算指标
#     MAE.append(getMAE(predictlabel, testlabel))
#     SMAPE.append(getSMAPE(predictlabel, testlabel))
#     RMSE.append(getRMSE(predictlabel, testlabel))
#     R2.append(getR2(predictlabel, testlabel))
#     # R.append(getR(predictlabel, testlabel))
#
#     RMSE = np.array(RMSE)
#     MAE = np.array(MAE)
#     SMAPE = np.array(SMAPE)
#     R2 = np.array(R2)
#     # R = np.array(R)
#     trainTime = np.array(trainTime)
#     testTime = np.array(testTime)
#
#     print("RMSE : ",RMSE)
#     print("R2 : ", R2)
#
#     datafram = {'RMSE': RMSE, 'MAE': MAE,'SMAPE': SMAPE, 'R2': R2, 'trainTime': trainTime, 'testTime': testTime, }
#     pd.DataFrame(datafram).to_csv(
#         'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\MSO\\GRU_huizong_test.csv', index=False)

#**************************************************************************************************************************************

import pandas as pd
import numpy as np
from keras.models import Sequential

import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import GRU, LSTM, Dropout, Dense
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

# #太阳黑子数量数据集
# name = 'Sunsplot'
#
# def data_process_slide(X, Y, X_sample_size, Y__sample_size, slide_size,iteration):
#     OUT = []
#     y = []
#     X.reshape(-1, 1)
#     Y.reshape(-1, 1)
#     for i in range(iteration):
#         OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
#         y.append(Y[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
#     OUT = np.array(OUT).reshape(iteration,X_sample_size)
#     y = np.array(y).reshape(iteration,Y__sample_size)
#     print("OUT.shape : ", OUT.shape)
#     print("y.shape : ", y.shape)
#     return OUT, y
#
# def generate_data_by_slide( X_sample_size, Y__sample_size, slide_size):
#     '''
#     # 前提是原始数据是一列数据，现在把它划分为（iteration,X_sample_size + Y__sample_size）
#     # :param X_sample_size: X的样本个数
#     # :param Y__sample_size: Y的样本个数
#     # :param slide_size: 窗口的滑动步长
#     # :return: None
#     '''
#     path = 'E:\yan_2\CFBLS_LCFBLS复现\dataset\Sunspot.csv'
#     X = pd.read_csv(path).values[1:2501, 2:3].reshape(2500, 1)
#
#     OUT = []
#     y = []
#
#     iteration = len(X) - X_sample_size - 1
#     for i in range(iteration):
#         OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
#         y.append(X[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
#     OUT = np.array(OUT).reshape(iteration,X_sample_size)
#     y = np.array(y).reshape(iteration,Y__sample_size)
#     print("OUT.shape : ", OUT.shape)
#     print("y.shape : ", y.shape)
#     data = np.hstack((OUT,y))
#
#     pd.DataFrame(data).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name,slide_size),
#                              index=False)
#
# def getDataFromCSV(path,Xcolumns,Ycolumns):
#     '''
#     # 将数据切片
#     # :return: 没有归一化的数据
#     '''
#     data = pd.read_csv(path).values[1:2401, : ]
#     trainLength = int(data.shape[0] * 0.8)
#
#     print("trainLength : ",trainLength)
#
#     traindata = data[:trainLength, :Xcolumns]
#     trainlabel = data[:trainLength, Xcolumns:Xcolumns+Ycolumns]
#     testdata = data[trainLength:, :Xcolumns]
#     testlabel = data[trainLength:, Xcolumns:Xcolumns+Ycolumns]
#
#     print("traindata.shape : ",traindata.shape)
#     print("trainlabel.shape : ", trainlabel.shape)
#     print("testdata.shape : ", testdata.shape)
#     print("testlabel.shape : ", testlabel.shape)
#
#     # pd.DataFrame(testlabel.reshape(-1,1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\Sunsplot\\test\\testReal.csv",index=False)
#     # pd.DataFrame(trainlabel.reshape(-1,1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\Sunsplot\\train\\trainReal.csv", index=False)
#
#     return traindata, trainlabel, testdata, testlabel

# name = 'MSO'
#
# def data_process_slide(X, Y, X_sample_size, Y__sample_size, slide_size,iteration):
#     OUT = []
#     y = []
#     X.reshape(-1, 1)
#     Y.reshape(-1, 1)
#     for i in range(iteration):
#         OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
#         y.append(Y[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
#     OUT = np.array(OUT).reshape(iteration,X_sample_size)
#     y = np.array(y).reshape(iteration,Y__sample_size)
#     print("OUT.shape : ", OUT.shape)
#     print("y.shape : ", y.shape)
#     return OUT, y
#
# def generate_data_by_slide( X_sample_size, Y__sample_size, slide_size):
#     '''
#     前提是原始数据是一列数据，现在把它划分为（iteration,X_sample_size + Y__sample_size）
#     :param X_sample_size: X的样本个数
#     :param Y__sample_size: Y的样本个数
#     :param slide_size: 窗口的滑动步长
#     :return: None
#     '''
#     path = 'E:\yan_2\CFBLS_LCFBLS复现\dataset\MSO_data_2.csv'
#     X = pd.read_csv(path)
#
#     OUT = []
#     y = []
#
#     iteration = (len(X) - X_sample_size - Y__sample_size)//slide_size
#     for i in range(iteration):
#         OUT.append(X[i*slide_size+1:i*slide_size+1+X_sample_size])
#         y.append(X[i*slide_size+1+X_sample_size:i*slide_size+1+X_sample_size+Y__sample_size])
#     OUT = np.array(OUT).reshape(iteration,X_sample_size)
#     y = np.array(y).reshape(iteration,Y__sample_size)
#     print("OUT.shape : ", OUT.shape)
#     print("y.shape : ", y.shape)
#     data = np.hstack((OUT,y))
#
#     pd.DataFrame(data).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name,slide_size),
#                              index=False)
#
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
#     # pd.DataFrame(testlabel.reshape(-1, 1)).to_csv(
#     #     "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO\\test\\testReal.csv", index=False)
#     # pd.DataFrame(trainlabel.reshape(-1, 1)).to_csv(
#     #     "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO\\train\\trainReal.csv", index=False)
#
#     print("traindata.shape : ",traindata.shape)
#     print("trainlabel.shape : ", trainlabel.shape)
#     print("testdata.shape : ", testdata.shape)
#     print("testlabel.shape : ", testlabel.shape)
#
#     return traindata, trainlabel, testdata, testlabel
#

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

    RMSE = []
    MAE = []
    SMAPE = []
    R2 = []
    R = []
    trainTime = []
    testTime = []

    traindata = traindata.reshape(-1, 6, 1)
    trainlabel = trainlabel.reshape(-1, 1)
    testdata = testdata.reshape(-1, 6, 1)
    testlabel = testlabel.reshape(-1, 1)

    model = Sequential()
    model.add(LSTM(32, input_shape=(traindata.shape[1], traindata.shape[2]), return_sequences=True,
                   name='LSTM_S0', kernel_initializer='orthogonal'))
    model.add(LSTM(64, return_sequences=False,
                   name='LSTM_S1'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    starttime = datetime.datetime.now()
    model.fit(traindata, trainlabel, epochs=50, batch_size=256, verbose=2, shuffle=False)
    endtime = datetime.datetime.now()
    trainTime = (endtime - starttime).total_seconds()

    starttime1 = datetime.datetime.now()
    predictlabel = model.predict(testdata).reshape(-1, 1)
    endtime1 = datetime.datetime.now()
    testTime = (endtime1 - starttime1).total_seconds()

    predictlabel = predictlabel.reshape(len(testdata), 1)
    testlabel = testlabel.reshape(-1, 1)

    endtime = datetime.datetime.now()

    pd.DataFrame(predictlabel.reshape(-1, 1)).to_csv(
        "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LSTM\\test\\LSTM_fangshan_predict.csv", index=False)

    # 计算指标
    MAE.append(getMAE(predictlabel, testlabel))
    SMAPE.append(getSMAPE(predictlabel, testlabel))
    RMSE.append(getRMSE(predictlabel, testlabel))
    R2.append(getR2(predictlabel, testlabel))
    # R.append(getR(predictlabel, testlabel))

    RMSE = np.array(RMSE)
    MAE = np.array(MAE)
    SMAPE = np.array(SMAPE)
    R2 = np.array(R2)
    # R = np.array(R)
    trainTime = np.array(trainTime)
    testTime = np.array(testTime)

    print("RMSE : ", RMSE)
    print("R2 : ", R2)

    datafram = {'RMSE': RMSE, 'MAE': MAE, 'SMAPE': SMAPE, 'R2': R2, 'trainTime': trainTime, 'testTime': testTime, }
    pd.DataFrame(datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\LSTM_huizong_test.csv', index=False)

    #MSO数据集实验结果
    # X_sample_size = 12
    # Y__sample_size = 1
    # slide_size = 1
    #
    # generate_data_by_slide(X_sample_size, Y__sample_size, slide_size)
    #
    # path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name, slide_size)
    # traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
    # initLength = 100
    #
    # RMSE = []
    # MAE = []
    # SMAPE = []
    # R2 = []
    # R = []
    # trainTime = []
    # testTime = []
    #
    # traindata = traindata.reshape(-1, 12, 1)
    # trainlabel = trainlabel.reshape(-1, 1)
    # testdata = testdata.reshape(-1, 12, 1)
    # testlabel = testlabel.reshape(-1, 1)
    #
    # model = Sequential()
    # model.add(GRU(32, input_shape=(traindata.shape[1], traindata.shape[2]), return_sequences=True,
    #                name='LSTM_S0', kernel_initializer='orthogonal'))
    # model.add(GRU(64, return_sequences=False,
    #                name='LSTM_S1'))
    # model.add(Dense(1))
    # model.compile(loss='mae', optimizer='adam')
    # print(model.summary())
    # starttime = datetime.datetime.now()
    # model.fit(traindata, trainlabel, epochs=100, batch_size=32, verbose=2, shuffle=False)
    # endtime = datetime.datetime.now()
    # trainTime = (endtime - starttime).total_seconds()
    #
    # starttime1 = datetime.datetime.now()
    # predictlabel = model.predict(testdata).reshape(-1, 1)
    # endtime1 = datetime.datetime.now()
    # testTime = (endtime1 - starttime1).total_seconds()
    #
    # predictlabel = predictlabel.reshape(len(testdata), 1)
    # testlabel = testlabel.reshape(-1, 1)
    #
    # endtime = datetime.datetime.now()
    #
    # pd.DataFrame(predictlabel.reshape(-1, 1)).to_csv(
    #     "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO\\GRU\\test\\LSTM_MSO_predict.csv", index=False)
    #
    # # 计算指标
    # MAE.append(getMAE(predictlabel, testlabel))
    # SMAPE.append(getSMAPE(predictlabel, testlabel))
    # RMSE.append(getRMSE(predictlabel, testlabel))
    # R2.append(getR2(predictlabel, testlabel))
    # # R.append(getR(predictlabel, testlabel))
    #
    # RMSE = np.array(RMSE)
    # MAE = np.array(MAE)
    # SMAPE = np.array(SMAPE)
    # R2 = np.array(R2)
    # # R = np.array(R)
    # trainTime = np.array(trainTime)
    # testTime = np.array(testTime)
    #
    # print("RMSE : ", RMSE)
    # print("R2 : ", R2)
    #
    # datafram = {'RMSE': RMSE, 'MAE': MAE, 'SMAPE': SMAPE, 'R2': R2, 'trainTime': trainTime, 'testTime': testTime, }
    # pd.DataFrame(datafram).to_csv(
    #     'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\MSO\\GRU_huizong_test.csv', index=False)

'''
    #太阳黑子数据集实验
    X_sample_size = 3
    Y__sample_size = 1
    slide_size = 1

    generate_data_by_slide(X_sample_size, Y__sample_size, slide_size)

    path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\{}\\{}_slide{}_data.csv".format(name, name,slide_size)
    traindata, trainlabel, testdata, testlabel = getDataFromCSV(path, X_sample_size, Y__sample_size)
    initLength = 100

    RMSE = []
    MAE = []
    SMAPE = []
    R2 = []
    R = []
    trainTime = []
    testTime = []

    traindata = traindata.reshape(-1, 3, 1)
    trainlabel = trainlabel.reshape(-1, 1)
    testdata = testdata.reshape(-1, 3, 1)
    testlabel = testlabel.reshape(-1, 1)

    model = Sequential()
    model.add(LSTM(32, input_shape=(traindata.shape[1], traindata.shape[2]), return_sequences=True,
                    name='LSTM_S0', kernel_initializer='orthogonal'))
    model.add(LSTM(64, return_sequences=False,
                    name='LSTM_S1'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    starttime = datetime.datetime.now()
    model.fit(traindata, trainlabel, epochs=100, batch_size=32, verbose=2, shuffle=False)
    endtime = datetime.datetime.now()
    trainTime = (endtime - starttime).total_seconds()

    starttime1 = datetime.datetime.now()
    predictlabel = model.predict(testdata).reshape(-1, 1)
    endtime1 = datetime.datetime.now()
    testTime = (endtime1 - starttime1).total_seconds()

    predictlabel = predictlabel.reshape(len(testdata), 1)
    testlabel = testlabel.reshape(-1,1)

    endtime = datetime.datetime.now()

    pd.DataFrame(predictlabel.reshape(-1, 1)).to_csv("E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO\\LSTM\\test\\LSTM_MSO_predict.csv",index=False)

    # 计算指标
    MAE.append(getMAE(predictlabel, testlabel))
    SMAPE.append(getSMAPE(predictlabel, testlabel))
    RMSE.append(getRMSE(predictlabel, testlabel))
    R2.append(getR2(predictlabel, testlabel))
    # R.append(getR(predictlabel, testlabel))

    RMSE = np.array(RMSE)
    MAE = np.array(MAE)
    SMAPE = np.array(SMAPE)
    R2 = np.array(R2)
    # R = np.array(R)
    trainTime = np.array(trainTime)
    testTime = np.array(testTime)

    print("RMSE : ",RMSE)
    print("R2 : ", R2)

    datafram = {'RMSE': RMSE, 'MAE': MAE,'SMAPE': SMAPE, 'R2': R2, 'trainTime': trainTime, 'testTime': testTime, }
    pd.DataFrame(datafram).to_csv(
        'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\MSO\\LSTM_huizong_test.csv', index=False)

'''
