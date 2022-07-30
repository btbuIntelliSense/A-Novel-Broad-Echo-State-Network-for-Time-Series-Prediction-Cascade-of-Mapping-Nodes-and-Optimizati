import logging
import pandas as pd
import numpy as np
from keras.layers import GRU, LSTM, Dropout, Dense
from keras.models import Sequential
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

name = "fangshan"

RMSE = []
MAE = []
SMAPE = []
R2 = []
R = []
trainTime = []
testTime = []

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
    name = 'fangshan'

    path = 'E:\\yan_1\\BLS_self\\fangshan.csv'
    data = pd.read_csv(path).values[1000:16000, :]

    label = data[:, :1].reshape(-1, 1)
    data = data[:, 1: 7]
    # print(data.shape, max(label) + 1)
    # traindata , testdata,trainlabel,testlabel = train_test_split(data,label,test_size=0.01,random_state = 0)

    initLen = 3000
    trainLen = 12000
    testLen = len(data) - trainLen

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    traindata = scaler1.fit_transform(data[initLen:trainLen, :])
    trainlabel = scaler2.fit_transform(label[initLen:trainLen])
    testdata = scaler1.fit_transform(data[trainLen: trainLen + testLen, :])
    testlabel = scaler2.fit_transform(label[trainLen: trainLen + testLen])

    #(batch_size, timesteps, input_dim)，即(训练数据量，时间步长，特征量)。因此不能直接把 [数据量*特征量]的二维矩阵输入，
    # 要用reshape进行转换。比如[50000,3]转化成时间步长为1的输入，即变成[50000,1,3]
    traindata = traindata.reshape(-1, 6, 1)
    trainlabel = trainlabel.reshape(-1, 1)
    testdata = testdata.reshape(-1, 6, 1)
    testlabel = testlabel.reshape(-1, 1)


    print(traindata.shape, trainlabel.shape, testdata.shape, testlabel.shape)

    model = Sequential()
    model.add(GRU(32, input_shape=(traindata.shape[1], traindata.shape[2]), return_sequences=True,
                    name='LSTM_S0', kernel_initializer='orthogonal'))
    model.add(GRU(64, return_sequences=False,
                    name='LSTM_S1'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    starttime = datetime.datetime.now()
    model.fit(traindata, trainlabel, epochs=100, batch_size=64, verbose=2, shuffle=False)
    endtime = datetime.datetime.now()
    trainTime = (endtime - starttime).total_seconds()

    starttime1 = datetime.datetime.now()
    predictlabel = model.predict(testdata).reshape(-1, 1)
    endtime1 = datetime.datetime.now()
    testTime = (endtime1 - starttime1).total_seconds()

    predictlabel = scaler2.inverse_transform(predictlabel).reshape(len(testdata), 1)
    testlabel = scaler2.inverse_transform(testlabel)

    endtime = datetime.datetime.now()

    '''
    
    
    log = logging.getLogger("main")  # 返回main对象
    log.setLevel(logging.DEBUG)  # 当seLevel设置为DEBUG时，可以截获取所有等级的输出
    # os.remove("VMD_Pm25.log")
    create_log = logging.FileHandler(
        "E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\log\\GRU_result_{}.log".format(name))  # 把日志记录在一个文件上
    create_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    create_log.setFormatter(formatter)
    log.addHandler(create_log)
    # Be careful with memory usage
    log.info('*****时间{}分割线******'.format(endtime))
    '''

    # 计算指标
    MAE.append(getMAE(predictlabel, testlabel))
    SMAPE.append(getSMAPE(predictlabel, testlabel))
    RMSE.append(getRMSE(predictlabel, testlabel))
    R2.append(getR2(predictlabel, testlabel))
    R.append(getR(predictlabel, testlabel))

    # print('*' * 100)
    # print('MAE : ', MAE)
    # print('SMAPE : ', SMAPE)
    # print('R2 : ', R2)
    # print('R  : ', R)
    print('RMSE : ', RMSE)
    # print('*' * 100)


    # log.info('          SMAPE为：{}'.format(SMAPE))
    # log.info('          MAE为：  {}'.format(MAE))
    # log.info('          RMSE为： {}'.format(RMSE))
    # log.info('          R2为：   {}'.format(R2))
    # log.info('          R 为：   {}'.format(R))
    # pd.DataFrame(predictlabel).to_csv('E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\csvFile\\{}\\GRU\\GRU_result.csv'.format(name), index=False)
    RMSE = np.array(RMSE)
    MAE = np.array(MAE)
    SMAPE = np.array(SMAPE)
    R2 = np.array(R2)
    R = np.array(R)
    trainTime = np.array(trainTime)
    testTime = np.array(testTime)


    # datafram = {'RMSE': RMSE, 'MAE': MAE,'SMAPE': SMAPE, 'R2': R2, 'R': R,'trainTime': trainTime, 'testTime': testTime, }
    # pd.DataFrame(datafram).to_csv(
    #     'E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\zhibiao\\GRU\\{}.csv'.format(
    #         name), index = False)

    # plt.figure(figsize=(13, 5))
    # plt.plot(testlabel, color='#7030A0', label='real')
    # # plt.plot(ESN_data, color='#00B0F0', label='ESN')
    # # plt.plot(BLS_data, color='#00B050', label='BLS')
    # plt.plot(predictlabel, color='#CD853F', label='GRU')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 把图例设置在外边
    # plt.ylabel('AQI')
    # plt.savefig('E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\picture\\GRU\\duibi.png', bbox_inches='tight', dpi=600)
    # plt.show()
