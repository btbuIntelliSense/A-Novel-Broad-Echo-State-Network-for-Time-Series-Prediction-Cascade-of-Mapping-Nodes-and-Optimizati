import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def plot_fangshan_line(startPos, endIndex):
    '''
    没有局部放大图的对比结果图
    :param startPos:
    :param endIndex:
    :return:
    '''
    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\fangshan\\test\\testReal.csv'
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_38_enhance_14.csv"
    # feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_32_enhance_16.csv"
    feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\自平稳优化指标体系实验结果数据\\test_4_实验画图.csv"

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    feipinwen_CFBLS_ESN_data = pd.read_csv(feipinwen_CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)

    plt.figure(figsize=(13, 5))
    x = np.arange(endIndex - startPos)
    plt.plot(x, real_data,color='#ed1941',label='real')
    plt.plot(x, CFBLS_ESN_data, color='#1d953f', label='CFBLS_ESN')
    plt.plot(x, feipinwen_CFBLS_ESN_data, color='#8552a1', label='CFBESN-NS')
    plt.ylabel('AQI')
    plt.xlabel('Time steps')
    plt.legend(loc='upper right')  # 把图例设置在外边
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\非平稳时序误差指标对比图.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_jubufangfa_fangshan_line():
    startPos = 1300
    endIndex = 1500
    MAX_EPISODES = endIndex - startPos
    x_axis_data = []
    for l in range(MAX_EPISODES):
        x_axis_data.append(l)

    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\fangshan\\test\\testReal.csv'
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_38_enhance_14.csv"
    # feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_32_enhance_16.csv"
    feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\自平稳优化指标体系实验结果数据\\test_4_实验画图.csv"

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    feipinwen_CFBLS_ESN_data = pd.read_csv(feipinwen_CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.plot(x_axis_data, real_data, color='#ed1941', label='real')
    plt.plot(x_axis_data, CFBLS_ESN_data, color='#1d953f', label='CMBESN')
    plt.plot(x_axis_data, feipinwen_CFBLS_ESN_data, color='#8552a1', label='CMBESN-OE')
    plt.ylabel('AQI', fontsize = 14)
    plt.xlabel('Time steps', fontsize = 14)

    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 把图例设置在外边
    plt.legend(loc='upper right', prop={'size' : 13})  # 把图例设置在外边
    # plt.ylabel('Death number')

    # 嵌入局部放大图的坐标系
    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.25, 0.62, 0.73, 1.2),
                       bbox_transform=ax.transAxes)
    # 在子坐标系中绘制原始数据

    axins.plot(x_axis_data, real_data, color='#ed1941', alpha=0.8)
    axins.plot(x_axis_data, CFBLS_ESN_data,  color='#1d953f', alpha=0.8)
    axins.plot(x_axis_data, feipinwen_CFBLS_ESN_data,  color='#8552a1', alpha=0.8)

    # 设置放大区间，调整子坐标系的显示范围
    # 设置放大区间
    zone_left = 115
    zone_right = 135

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.0  # x轴显示范围的扩展比例
    y_ratio = 0.07  # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = x_axis_data[zone_left] - (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio
    xlim1 = x_axis_data[zone_right] + (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio

    # Y轴的显示范围
    y = np.hstack((real_data[zone_left:zone_right], CFBLS_ESN_data[zone_left:zone_right],
                   feipinwen_CFBLS_ESN_data[zone_left:zone_right]))

    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 建立父坐标系与子坐标系的连接线
    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "#CD853F")

    print('xlim0 : ', xlim0)
    print('xlim1 : ', xlim1)
    print('ylim0 : ', ylim0)
    print('ylim1 : ', ylim1)
    # 画两条线
    xy = (xlim0, ylim0)  # (60,-409)
    xy2 = (xlim0, ylim0)  # (60,409)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, color='#b22c46')
    axins.add_artist(con)

    xy = (xlim0, ylim1)  # (90,-409)
    xy2 = (xlim1, ylim1)  # (90,-409)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, color='#b22c46')
    axins.add_artist(con)

    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\第一版修改的图\\Figure 17.svg', format='svg', bbox_inches='tight', dpi=800)  # 高清图
    plt.show()

def plot_fangshan_feipinwen_RMSE_AND_Regularization_coefficient():
    data_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\自平稳优化指标体系数据汇总-补加的正则化系数.csv"
    RMSE_data = pd.read_csv(data_path).values[2:37, :1]
    RC_data = pd.read_csv(data_path).values[2:37, 6:7]
    x = np.arange(RMSE_data.shape[0])
    plt.figure(figsize=(13, 5))
    plt.subplot(211)
    plt.plot(x, RMSE_data, color='#6950a1')
    plt.ylabel('RMSE', fontsize = 14)
    # plt.xlabel('Time steps')

    plt.subplot(212)
    plt.plot(x, RC_data, color='#6950a1')
    plt.ylabel('λ', fontsize = 14)
    plt.xlabel('Time steps', fontsize = 14)
    plt.yscale('log')  # 设置纵坐标的缩放
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\CM实验结果图\\CM_非平稳时序误差指标RMSE与λ_科学计数法.svg', format='svg', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()


def plot_RMSE():
    data_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\自平稳优化指标体系数据汇总-补加的正则化系数.csv"
    RMSE_data = pd.read_csv(data_path).values[2:38, :1]
    x = np.arange(RMSE_data.shape[0])
    plt.figure(figsize=(13, 5))
    plt.plot(x, RMSE_data, color='#6950a1')
    plt.ylabel('RMSE')
    plt.xlabel('Time steps')
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\非平稳时序误差指标中RMSE变化.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_M():
    data_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\自平稳优化指标体系数据汇总-补加的正则化系数.csv"
    RMSE_data = pd.read_csv(data_path).values[2:38, 8:9]
    x = np.arange(RMSE_data.shape[0])
    plt.figure(figsize=(13, 5))
    plt.plot(x, RMSE_data, color='#6950a1')
    plt.axhline(0.995, color='r', linestyle='--')

    plt.ylabel('M')
    plt.xlabel('Time steps')
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\非平稳时序误差指标中M变化.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_S():
    data_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\自平稳优化指标体系数据汇总.csv"
    RMSE_data = pd.read_csv(data_path).values[:, 9:10]
    x = np.arange(RMSE_data.shape[0])
    plt.figure(figsize=(13, 5))
    plt.plot(x, RMSE_data, color='#6950a1')
    plt.ylabel('S')
    plt.xlabel('Time steps')
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\非平稳时序误差指标中S变化.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_Regularization_coefficient():
    path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\fangshan\\自平稳优化指标体系数据汇总-补加的正则化系数.csv"
    data = pd.read_csv(path).values[2:38, 6:7]
    x = np.arange(data.shape[0])
    print(x)
    plt.figure(figsize=(13, 5))
    plt.plot(x, data, color='#6950a1')
    plt.ylabel('λ')
    plt.xlabel('Time steps')
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\非平稳时序误差指标中λ变化.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_jubufangfa_MSO_8_line():
    '''
    绘制了预测曲线图和误差对比图，两个图哦
    :return:
    '''
    startPos = 1000
    endIndex = 1050
    MAX_EPISODES = endIndex - startPos
    x_axis_data = []
    for l in range(MAX_EPISODES):
        x_axis_data.append(l)

    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv'
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_34_enhance_24.csv"
    # feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_32_enhance_16.csv"
    feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\自平稳优化指标体系实验结果数据\\调试版\\test\\test_79_补.csv"

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    feipinwen_CFBLS_ESN_data = pd.read_csv(feipinwen_CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    plt.plot(x_axis_data, real_data, color='#ed1941', label='real')
    plt.plot(x_axis_data, CFBLS_ESN_data, color='#1d953f', label='CMBESN')
    plt.plot(x_axis_data, feipinwen_CFBLS_ESN_data, color='#8552a1', label='CMBESN-OE')
    plt.ylabel('MSO Value', fontsize = 14)
    plt.xlabel('Time steps', fontsize = 14)

    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 把图例设置在外边
    plt.legend(loc='upper left', prop={'size' : 13})  # 把图例设置在外边
    # plt.ylabel('Death number')

    # 嵌入局部放大图的坐标系
    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.14, 0.09, 0.55, 1.2),
                       bbox_transform=ax.transAxes)
    # 在子坐标系中绘制原始数据

    axins.plot(x_axis_data, real_data, color='#ed1941', alpha=0.8)
    axins.plot(x_axis_data, CFBLS_ESN_data, color='#1d953f', alpha=0.8)
    axins.plot(x_axis_data, feipinwen_CFBLS_ESN_data, color='#8552a1', alpha=0.8)

    # 设置放大区间，调整子坐标系的显示范围
    # 设置放大区间
    zone_left = 6
    zone_right = 12

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.0  # x轴显示范围的扩展比例
    y_ratio = 0.07  # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = x_axis_data[zone_left] - (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio
    xlim1 = x_axis_data[zone_right] + (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio

    # Y轴的显示范围
    y = np.hstack((real_data[zone_left:zone_right], CFBLS_ESN_data[zone_left:zone_right],
                   feipinwen_CFBLS_ESN_data[zone_left:zone_right]))

    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 建立父坐标系与子坐标系的连接线
    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "#CD853F")

    print('xlim0 : ', xlim0)
    print('xlim1 : ', xlim1)
    print('ylim0 : ', ylim0)
    print('ylim1 : ', ylim1)
    # 画两条线
    xy = (xlim0, ylim0)  # (60,-409)
    xy2 = (xlim0, ylim1)  # (60,409)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, color='#b22c46')
    axins.add_artist(con)

    xy = (xlim1, ylim0)  # (90,-409)
    xy2 = (xlim1, ylim1)  # (90,-409)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, color='#b22c46')
    axins.add_artist(con)

    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\CM_非平稳指标使用对比的局部放大图_折线图.svg', format='svg',bbox_inches='tight', dpi=800)  # 高清图
    plt.show()

    startPos1 = 1700
    endIndex1 = 2000
    real_data_1 = pd.read_csv(real_path).values[startPos1:endIndex1].reshape(-1, 1)
    CFBLS_ESN_data_1 = pd.read_csv(CFBLS_ESN_path).values[startPos1:endIndex1].reshape(-1, 1)
    feipinwen_CFBLS_ESN_data_1 = pd.read_csv(feipinwen_CFBLS_ESN_path).values[startPos1:endIndex1].reshape(-1, 1)

    x1 = np.arange(endIndex1 - startPos1)
    fig1, vax = plt.subplots(1, 1, figsize=(11, 4))
    ERROR_qian = []
    ERROR_hou = []

    for i in range(endIndex1 - startPos1):
        ERROR_qian.append(CFBLS_ESN_data_1[i] - real_data_1[i])
        ERROR_hou.append(feipinwen_CFBLS_ESN_data_1[i] - real_data_1[i])

    ERROR_qian = np.array(ERROR_qian).reshape(-1, 1)
    ERROR_hou = np.array(ERROR_hou).reshape(-1, 1)

    plt.vlines(x1, [0], ERROR_qian, colors='#00a6ac', label='CMBESN')
    plt.vlines(x1, [0], ERROR_hou, colors='#9b95c9', label='CMBESN-OE')

    # plt.plot(x, ERROR_ESN, color='#b2d235', label='ESN')
    # plt.plot(x, ERROR_BLS, color='#f47920', label='BLS')
    # plt.plot(x, ERROR_CFBLS, color='#817936', label='CFESN')
    # plt.plot(x, ERROR_CFBESN, color='#8552a1', label='CFBESN')
    # plt.plot(x, ERROR_GRU, color='#009ad6', label='GRU')
    plt.ylabel('Errors')
    plt.xlabel('Time steps')
    # fig1.subplots_adjust(right=0.88)  # 调整边距和子图的间距
    plt.legend(loc='upper right')  # 把图例设置在外边
    # plt.legend(loc='upper left', bbox_to_anchor=(0.995, 1))  # 把图例设置在外边
    # fig.subplots_adjust(right=1.5)  # 调整边距和子图的间距
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\CM实验结果图\\CM_误差垂线图.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_ERROR_MSO_8_scatter():

    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv'
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_34_enhance_24.csv"
    # feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_32_enhance_16.csv"
    feipinwen_CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\自平稳优化指标体系实验结果数据\\调试版\\test\\test_79_补.csv"

    startPos1 = 1695
    endIndex1 = 1725
    real_data_1 = pd.read_csv(real_path).values[startPos1:endIndex1].reshape(-1, 1)
    CFBLS_ESN_data_1 = pd.read_csv(CFBLS_ESN_path).values[startPos1:endIndex1].reshape(-1, 1)
    feipinwen_CFBLS_ESN_data_1 = pd.read_csv(feipinwen_CFBLS_ESN_path).values[startPos1:endIndex1].reshape(-1, 1)

    x1 = np.arange(endIndex1 - startPos1)
    fig1, vax = plt.subplots(1, 1, figsize=(11, 5))
    ERROR_qian = []
    ERROR_hou = []

    for i in range(endIndex1 - startPos1):
        ERROR_qian.append(CFBLS_ESN_data_1[i] - real_data_1[i])
        ERROR_hou.append(feipinwen_CFBLS_ESN_data_1[i] - real_data_1[i])

    ERROR_qian = np.array(ERROR_qian).reshape(-1, 1)
    ERROR_hou = np.array(ERROR_hou).reshape(-1, 1)
    lim_xy_max = max(max(ERROR_qian), max(ERROR_hou)) + 0.05
    plt.scatter(x1, ERROR_qian, s=70, marker='.', c='#d71345', label='CMBESN')
    plt.scatter(x1, ERROR_hou, s=70, marker='.', c='#494e8f', label='CMBESN-OE')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 11})

    # plt.plot(x, ERROR_ESN, color='#b2d235', label='ESN')
    # plt.plot(x, ERROR_BLS, color='#f47920', label='BLS')
    # plt.plot(x, ERROR_CFBLS, color='#817936', label='CFESN')
    # plt.plot(x, ERROR_CFBESN, color='#8552a1', label='CFBESN')
    # plt.plot(x, ERROR_GRU, color='#009ad6', label='GRU')
    plt.ylabel('Errors')
    plt.xlabel('Time steps')
    # fig1.subplots_adjust(right=0.88)  # 调整边距和子图的间距
    plt.legend(loc='upper right', prop={'size': 13})  # 把图例设置在外边
    # plt.legend(loc='upper left', bbox_to_anchor=(0.995, 1))  # 把图例设置在外边
    # fig.subplots_adjust(right=1.5)  # 调整边距和子图的间距
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\CM_误差垂线图.svg', format='svg', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_MSO_8_feipinwen_RMSE_AND_Regularization_coefficient():
    data_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\zhibiao\\huizong\\MSO_8\\自平稳优化指标体系数据汇总-调试版_补.csv"
    RMSE_data = pd.read_csv(data_path).values[26:55, :1]
    RC_data = pd.read_csv(data_path).values[26:55, 6:7]
    x = np.arange(RMSE_data.shape[0])
    plt.figure(figsize=(12, 5))
    plt.subplot(211)
    plt.plot(x, RMSE_data, color='#6950a1')
    plt.ylabel('RMSE', fontsize=14)
    # plt.xlabel('Time steps')

    plt.subplot(212)
    plt.plot(x, RC_data, color='#6950a1')
    plt.ylabel('λ', fontsize=14)
    plt.xlabel('Time steps', fontsize=14)
    plt.yscale('log')  # 设置纵坐标的缩放
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\CM实验结果图\\CM_非平稳时序误差指标RMSE与λ_科学计数法.svg', format='svg', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

if __name__ == '__main__':

    # plot_fangshan_line(1000,1800)#前段还可以
    # plot_Regularization_factor()
    # plot_RMSE()
    # plot_M()
    # plot_S()
    # plot_Regularization_coefficient()
    # plot_fangshan_line(1000,1800)
    # plot_fangshan_feipinwen_RMSE_AND_Regularization_coefficient()
    # plot_fangshan_line(1000,1800)
    # plot_jubufangfa_fangshan_line()
    # plot_jubufangfa_MSO_8_line()
    # plot_MSO_8_feipinwen_RMSE_AND_Regularization_coefficient()

    """
    修改意见第一版
    """
    # plot_jubufangfa_MSO_8_line()
    # plot_ERROR_MSO_8_scatter()
    # plot_jubufangfa_fangshan_line()
    # plot_fangshan_feipinwen_RMSE_AND_Regularization_coefficient()
    plot_MSO_8_feipinwen_RMSE_AND_Regularization_coefficient()