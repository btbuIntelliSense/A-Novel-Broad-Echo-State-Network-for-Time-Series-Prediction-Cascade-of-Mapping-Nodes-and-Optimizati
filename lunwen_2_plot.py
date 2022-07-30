import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
# import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['FangSong']
# mpl.rcParams['axes.unicode_minus']=False
from matplotlib import cm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import xlrd


def zhexiantu_fangshan(mappingNumber, enhanceNumber):
    '''
    绘制折线图
    :return:
    '''
    fig = plt.figure(figsize=(13, 5))
    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\fangshan\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\ESN\\test\\ESN_resSize_450.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(mappingNumber,enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS\\test\\LCFBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    # CEBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CEBLS\\test\\CEBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS_ESN\\test\\LCFBLS_ESN_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\GRU\\test\\GRU_fangshan_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LSTM\\test\\LSTM_fangshan_predict.csv"

    # GRU_path = 'E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\csvFile\\fangshan\\GRU\\GRU_result.csv'
    # LSTM_path = 'E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\csvFile\\fangshan\\LSTM\\LSTM_result.csv'

    startPos = 788
    endIndex = 858
    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1,1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    # LCFBLS_data = pd.read_csv(LCFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    # CEBLS_data = pd.read_csv(CEBLS_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    # LCFBLS_ESN_data = pd.read_csv(LCFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(-1, 1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(-1, 1)

    color = ['#7030A0', '#00B0F0', '#CD853F', '#00B050', '#92D050', '#FFFF00', '#FFC000', '#9DC3E6', '#8B0000', '#FF0000']
    x = np.arange(endIndex-startPos)

    plt.plot(x, real_data, color='#d71345', label='real')
    plt.plot(x, BLS_data, color='#f47920', label='BLS')
    plt.plot(x, ESN_data, color='#b2d235', label='ESN')
    plt.plot(x, CFBLS_data, color='#8552a1', label='CMBLS')
    # plt.plot(x, BLS_ESN_data, color='#005344', label='BLS_ESN')
    plt.plot(x, CFBLS_ESN_data, color='#ef5b9c', label='GRU')
    plt.plot(x, GRU_data, color='#009ad6', label='CMBESN')
    # plt.plot(x, LSTM_data, color='#ea66a6', label='LSTM')


    # plt.title("Mapping node:{} Enhance node:{}".format(mappingNumber,enhanceNumber))
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) #把图例设置在外边
    plt.legend(loc='lower left', prop={'size': 12})  # 把图例设置在外边
    plt.ylabel('AQI', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)

    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\第一版修改的图\\ Figure 10.svg', format='svg',
                bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_ERROR_fangshan(mappingNumber, enhanceNumber):

    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\fangshan\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\ESN\\test\\ESN_resSize_450.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS\\test\\LCFBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    # CEBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CEBLS\\test\\CEBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS_ESN\\test\\LCFBLS_ESN_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\GRU\\test\\GRU_fangshan_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LSTM\\test\\LSTM_fangshan_predict.csv"

    # GRU_path = 'E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\csvFile\\fangshan\\GRU\\GRU_result.csv'
    # LSTM_path = 'E:\\yan_1\\BLS_self\\真正意义上的BLS和ESN结合(可以用来实验)\\备份好的BLS和增加增强节点的BLS\\result\\csvFile\\fangshan\\LSTM\\LSTM_result.csv'

    startPos = 300
    endIndex = 380
    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1, 1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    # LCFBLS_data = pd.read_csv(LCFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    # CEBLS_data = pd.read_csv(CEBLS_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    # LCFBLS_ESN_data = pd.read_csv(LCFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(-1, 1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(-1, 1)

    # plt.plot(x, real_data,'-*',color='#ef5b9c',label='real')
    # plt.plot(x, BLS_data,'-3', color='#840228', label='BLS')
    # plt.plot(x, ESN_data, '-2', color='#1d953f', label='ESN')
    # plt.plot(x, CFBLS_data,'-+', color='#87843b', label='CFBLS')
    # # plt.plot(x, LCFBLS_data, color='#d71345', label='LCFBLS')
    # # plt.plot(x, CEBLS_data, color='#bed742', label='CEBLS')
    # plt.plot(x, BLS_ESN_data,'-.', color='#9b95c9', label='BLS_ESN')
    # plt.plot(x, CFBLS_ESN_data,'-s', color='#005344', label='CFBLS_ESN')
    # # plt.plot(x, LCFBLS_ESN_data, color='#1d953f', label='LCFBLS_ESN')
    # plt.plot(x, GRU_data, '-.x',color='#ffc20e', label='GRU')
    # plt.plot(x, LSTM_data, '-.1', color='#8552a1', label='LSTM')
    # ed1941 b2d235 ba8448 2b4490 5c7a29 009ad6 2585a6 585eaa ea66a6

    x1 = np.arange(endIndex - startPos)
    ERROR_ESN = []
    ERROR_BLS = []
    ERROR_CFBLS = []
    ERROR_CFBESN = []
    ERROR_GRU = []

    for i in range(endIndex - startPos):
        ERROR_ESN.append(abs(ESN_data[i] - real_data[i]))
        ERROR_BLS.append(abs(BLS_data[i] - real_data[i]))
        ERROR_CFBLS.append(abs(CFBLS_data[i] - real_data[i]))
        ERROR_CFBESN.append(abs(CFBLS_ESN_data[i] - real_data[i]))
        ERROR_GRU.append(abs(GRU_data[i] - real_data[i]))

    ERROR_ESN = np.array(ERROR_ESN).reshape(-1, 1)
    ERROR_BLS = np.array(ERROR_BLS).reshape(-1, 1)
    ERROR_CFBLS = np.array(ERROR_CFBLS).reshape(-1, 1)
    ERROR_CFBESN = np.array(ERROR_CFBESN).reshape(-1, 1)
    ERROR_GRU = np.array(ERROR_GRU).reshape(-1, 1)

    gs = gridspec.GridSpec(2, 6) #创建2行6列的网格

    lim_xy_max = max(max(ERROR_ESN), max(ERROR_BLS), max(ERROR_CFBLS), max(ERROR_CFBESN), max(ERROR_GRU)) + 0.1

    plt.figure(figsize=(12, 6))
    plt.subplot(gs[0, :2]) #gs[哪一行，列的范围]
    gs.update(hspace=0.3, wspace=0.8)

    plt.scatter(x1, ERROR_ESN, s=40, marker='.', c='#00a6ac', label='ESN')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 11})

    plt.subplot(gs[0, 2:4]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_BLS, s=40, marker='.', c='#9b95c9', label='BLS')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 11})

    plt.subplot(gs[0, 4:6]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_CFBLS, s=40, marker='.', c='#fdb933', label='CMBLS')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 11})

    plt.subplot(gs[1, 1:3]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_CFBESN, s=40, marker='.', c='#a3cf62', label='GRU')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 12})

    plt.subplot(gs[1, 3:5]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_GRU, s=40, marker='.', c='#f05b72', label='CMBESN')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 12})


    # plt.subplots_adjust(hspace=0.3,wspace=0.5)  # 调节上下子图之间的行距

    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\第一版修改的图\\Figure 12.svg', format='svg',
                            bbox_inches='tight', dpi=600)  # 高清图
    plt.show()


def fangshan_boxplot(mappingNumber, enhanceNumber):
    '''
    这个是把多个箱线图画在一个坐标系中的程序,这里实现的是将所有的预测数据用箱线图来展示出来
    :return:

    '''

    fig = plt.figure(figsize=(13, 5))
    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\fangshan\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\ESN\\test\\ESN_resSize_450.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS\\test\\LCFBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    # CEBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CEBLS\\test\\CEBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS_ESN\\test\\LCFBLS_ESN_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\GRU\\test\\GRU_fangshan_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LSTM\\test\\LSTM_fangshan_predict.csv"


    startPos = 100
    endIndex = 3000

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(1, -1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(1, -1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(1, -1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(1, -1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(1, -1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(1, -1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(1, -1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(1, -1)


    SAMPE = {'REAL':real_data[0],'CMBESN':CFBLS_ESN_data[0],'BLS':BLS_data[0],
             'ESN':ESN_data[0],'CMBLS':CFBLS_data[0],'GRU':GRU_data[0]}

    df = pd.DataFrame(SAMPE)

    ff = df.boxplot(showmeans=True, return_type='dict', showfliers=False, vert=True,
                   sym='o', widths=0.3, whis=True, patch_artist=False, meanline=False,
                   showcaps=True,flierprops={'markeredgecolor': 'k', 'markersize': 8},
                   meanprops={'markeredgecolor': 'k', 'markerfacecolor': 'k', 'marker': '*'},figsize=(13, 5))

    # color = ['#7030A0', '#00B0F0', '#CD853F', '#00B050', '#92D050', '#FFFF00', '#FFC000', '#9DC3E6', '#8B0000',
    #          '#FF0000']

    # 这里共有5个box
    # color = ['#6F933B', '#1B459C', '#16A2FF', '#CE7937', '#FF4025','#7030A0','#9DC3E6']  # 有多少box就对应设置多少颜色
    # colors是为了解决上下须不同步，上下帽不同步颜色设置的
    colors = ['#6F933B', '#6F933B', '#1B459C', '#1B459C', '#16A2FF', '#16A2FF', '#CE7937',
              '#CE7937', '#FF4025', '#FF4025', '#7030A0', '#7030A0', '#9DC3E6', '#9DC3E6']

    # ed1941 b2d235 ba8448 2b4490 5c7a29 009ad6 2585a6 585eaa ea66a6
    newColor = ['#ef5b9c', '#f47920', '#b2d235', '#817936', '#f15a22', '#009ad6', '#8552a1', '#ffc20e']
    newColors = ['#ef5b9c', '#ef5b9c', '#f47920', '#f47920', '#b2d235', '#b2d235', '#817936',
                '#817936', '#f15a22', '#f15a22', '#009ad6', '#009ad6', '#8552a1', '#8552a1',
                 '#ffc20e', '#ffc20e']

    # fig = plt.figure(figsize=(13, 5))
    linewidth = 1.8
    for box, c in zip(ff['boxes'], newColor):
        # 箱体边框颜色
        box.set(color=c, linewidth=linewidth)
        # 箱体内部填充颜色
        # box.set(facecolor=c)

    # 这里设置的是各个box的其他属性
    for whisker, c in zip(ff['whiskers'], newColors):
        whisker.set(color=c, linewidth=linewidth)
    for cap, c in zip(ff['caps'], newColors):
        cap.set(color=c, linewidth=linewidth)

    for median, c in zip(ff['medians'], newColor):
        median.set(color=c, linewidth=linewidth)
    for mean, color in zip(ff['means'], newColor):  # 星花是均值
        mean.set(markeredgecolor=color, markerfacecolor=color, linewidth=1)
    for flier, c in zip(ff['fliers'], newColor):
        flier.set(marker='o', color=c, alpha=0.5)
    plt.ylabel('AQI')
    # plt.savefig('result/SMAPE.png', bbox_inches='tight',dpi=600)  # 高清图
    # fig.subplots_adjust(right=0.82)  # 调整边距和子图的间距
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\CM实验结果图\\CM_箱线图.png', bbox_inches='tight', dpi=600)
    plt.show()


def plot_3D_weipinwen_fangshan():
    '''
    没有使用自平稳优化指标体系的3D绘图
    :return:
    '''
    data_path = r'E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\未使用自平稳指标体系的3D图.xlsx'
    data = pd.read_excel(data_path).values[:,:]
    # print(data)
    X = np.linspace(12,20,5,dtype=int)
    Y = np.linspace(20,38,10,dtype=int)
    z = np.array(data[:, 3:4]).reshape(10,5)
    print(Y)

    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(13, 8))
    ax = plt.axes(projection='3d')
    # ax.plot_wireframe(X, Y, z, rstride = 1, cstride = 1, cmap='RdPu')
    ax.plot_surface(X, Y, z, rstride=1, cstride=1, alpha=0.3,cmap="hsv", linewidth=0, antialiased=False)

    ax.contour(X, Y, z, zdir='x', offset=11, cmap="hsv_r")
    # ax.contour(X, Y, z, zdir='z', offset=0, cmap="hsv_r")

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

    ax.set_xlabel('mapping node')
    ax.set_ylabel('ESN')
    ax.set_zlabel('RMSE', rotation=90)
    # ax.grid(False, linestyle = "-.", color = "red", linewidth = "1")

    ax.view_init(elev=19, azim=35)


    # ax.contour(X, Y, z, cmap=cm.coolwarm)
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\未使用自平稳指标体系的3D图.png', bbox_inches='tight', dpi=600)

    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    plt.show()

def plot_3D_pinwen_fangshan():
    '''
    没有使用自平稳优化指标体系的3D绘图
    :return:
    '''
    data_path = r'E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\自平稳指标体系的3D图.xlsx'
    data = pd.read_excel(data_path).values[:,:]
    # print(data)
    X = np.linspace(12,28,9,dtype=int)
    Y = np.linspace(20,38,10,dtype=int)
    z = np.array(data[:, 3:4]).reshape(10,9)

    '''
        cmap = "以下的取值"
        'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 
        'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 
        'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 
        'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 
        'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 
        'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 
        'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 
        'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 
        'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 
        'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
        'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 
        'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 
        'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 
        'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

        '''

    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(13, 8))
    ax = plt.axes(projection='3d')
    # ax.plot_wireframe(X, Y, z, rstride = 1, cstride = 1, cmap='RdPu')
    ax.plot_surface(X, Y, z, rstride=2, cstride=2, alpha=0.45,cmap="Paired", linewidth=0, antialiased=False)

    ax.contour(X, Y, z, zdir='x', offset=11, cmap="Paired")
    # ax.contour(X, Y, z, zdir='y', offset=20, cmap="Paired")

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    ax.set_xlabel('mapping node')
    ax.set_ylabel('ESN')
    ax.set_zlabel('RMSE', rotation=90)
    # ax.grid(False, linestyle = "-.", color = "red", linewidth = "1")

    ax.view_init(elev=17, azim=42)


    # ax.contour(X, Y, z, cmap=cm.coolwarm)
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\使用自平稳指标体系的3D图.png', bbox_inches='tight', dpi=600)

    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    plt.show()


def getStationaryFigure():
    #得到自平稳指标体系RMSE曲线图
    StationaryPath = "E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\自平稳指标体系实验数据.xlsx"
    StationaryData = pd.read_excel(r'E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\自平稳指标体系实验数据.xlsx').values[1:].reshape(-1,1)
    fig = plt.figure(figsize=(13, 5))
    x = np.arange(len(StationaryData))
    plt.plot(x, StationaryData, 'c-*',color='#1B459C')
    plt.show()

def zhexiantu_MSO_8(mappingNumber, enhanceNumber):
    '''
    绘制折线图
    :return:
    '''
    fig = plt.figure(figsize=(13, 5))
    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\ESN\\test\\ESN_resSize_920.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(mappingNumber,enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS\\test\\LCFBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    # CEBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CEBLS\\test\\CEBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS_ESN\\test\\LCFBLS_ESN_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\GRU\\test\\GRU_MSO_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\LSTM\\test\\LSTM_MSO_predict.csv"

    startPos = 1000
    endIndex = 1601
    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1,1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    # LCFBLS_data = pd.read_csv(LCFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    # CEBLS_data = pd.read_csv(CEBLS_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    # LCFBLS_ESN_data = pd.read_csv(LCFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(-1, 1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(-1, 1)

    color = ['#7030A0', '#00B0F0', '#CD853F', '#00B050', '#92D050', '#FFFF00', '#FFC000', '#9DC3E6', '#8B0000', '#FF0000']
    x = np.arange(endIndex-startPos)
    plt.plot(x, real_data,color='#ef5b9c',label='real')
    plt.plot(x, BLS_data, color='#840228', label='BLS')
    plt.plot(x, ESN_data, color='#1d953f', label='ESN')
    plt.plot(x, CFBLS_data, color='#87843b', label='CFBLS')
    # plt.plot(x, BLS_ESN_data, color='#9b95c9', label='BLS_ESN')
    plt.plot(x, CFBLS_ESN_data, color='#005344', label='CFBLS_ESN')
    plt.plot(x, GRU_data, color='#ffc20e', label='GRU')
    # plt.plot(x, LSTM_data, color='#8552a1', label='LSTM')
    #ed1941 b2d235 ba8448 2b4490 5c7a29 009ad6 2585a6 585eaa ea66a6

    # plt.title("Mapping node:{} Enhance node:{}".format(mappingNumber,enhanceNumber))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) #把图例设置在外边
    plt.ylabel('AQI')
    fig.subplots_adjust(right=0.90) #调整边距和子图的间距
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\折线图.png', bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def MSO_8_boxplot(mappingNumber, enhanceNumber):
    '''
    这个是把多个箱线图画在一个坐标系中的程序,这里实现的是将所有的预测数据用箱线图来展示出来
    :return:
    '''

    fig = plt.figure(figsize=(13, 5))
    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\ESN\\test\\ESN_resSize_920.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS\\test\\LCFBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    # CEBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\CEBLS\\test\\CEBLS_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    # LCFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\fangshan\\LCFBLS_ESN\\test\\LCFBLS_ESN_mapping_{}_enhance_{}.csv".format(
    #     mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\GRU\\test\\GRU_MSO_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\LSTM\\test\\LSTM_MSO_predict.csv"


    startPos = 100
    endIndex = 3000

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(1, -1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(1, -1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(1, -1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(1, -1)
    # LCFBLS_data = pd.read_csv(LCFBLS_path).values[startPos:endIndex].reshape(1, -1)
    # CEBLS_data = pd.read_csv(CEBLS_path).values[startPos:endIndex].reshape(1, -1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(1, -1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(1, -1)
    # LCFBLS_ESN_data = pd.read_csv(LCFBLS_ESN_path).values[startPos:endIndex].reshape(1, -1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(1, -1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(1, -1)


    SAMPE = {'REAL':real_data[0],'CFBESN':CFBLS_ESN_data[0],'BESN':BLS_ESN_data[0],'BLS':BLS_data[0],
             'ESN':ESN_data[0],'CFBLS':CFBLS_data[0],'GRU':GRU_data[0],'LSTM':LSTM_data[0]}

    df = pd.DataFrame(SAMPE)

    ff = df.boxplot(showmeans=True, return_type='dict', showfliers=False, vert=True,
                   sym='o', widths=0.3, whis=True, patch_artist=False, meanline=False,
                   showcaps=True,flierprops={'markeredgecolor': 'k', 'markersize': 8},
                   meanprops={'markeredgecolor': 'k', 'markerfacecolor': 'k', 'marker': '*'},figsize=(13, 5))

    # color = ['#7030A0', '#00B0F0', '#CD853F', '#00B050', '#92D050', '#FFFF00', '#FFC000', '#9DC3E6', '#8B0000',
    #          '#FF0000']

    # 这里共有5个box
    # color = ['#6F933B', '#1B459C', '#16A2FF', '#CE7937', '#FF4025','#7030A0','#9DC3E6']  # 有多少box就对应设置多少颜色
    # colors是为了解决上下须不同步，上下帽不同步颜色设置的
    colors = ['#6F933B', '#6F933B', '#1B459C', '#1B459C', '#16A2FF', '#16A2FF', '#CE7937',
              '#CE7937', '#FF4025', '#FF4025', '#7030A0', '#7030A0', '#9DC3E6', '#9DC3E6']

    # ed1941 b2d235 ba8448 2b4490 5c7a29 009ad6 2585a6 585eaa ea66a6
    newColor = ['#ed1941', '#b2d235', '#ba8448', '#2b4490', '#5c7a29', '#009ad6', '#2585a6','#585eaa','#ea66a6']
    newColors = ['#ed1941', '#ed1941', '#b2d235', '#b2d235', '#ba8448', '#ba8448', '#2b4490',
                '#2b4490', '#5c7a29', '#5c7a29', '#009ad6', '#009ad6', '#2585a6', '#2585a6',
                 '#585eaa', '#585eaa', '#ea66a6', '#ea66a6']

    # fig = plt.figure(figsize=(13, 5))
    linewidth = 1.8
    for box, c in zip(ff['boxes'], newColor):
        # 箱体边框颜色
        box.set(color=c, linewidth=linewidth)
        # 箱体内部填充颜色
        # box.set(facecolor=c)

    # 这里设置的是各个box的其他属性
    for whisker, c in zip(ff['whiskers'], newColors):
        whisker.set(color=c, linewidth=linewidth)
    for cap, c in zip(ff['caps'], newColors):
        cap.set(color=c, linewidth=linewidth)

    for median, c in zip(ff['medians'], newColor):
        median.set(color=c, linewidth=linewidth)
    for mean, color in zip(ff['means'], newColor):  # 星花是均值
        mean.set(markeredgecolor=color, markerfacecolor=color, linewidth=1)
    for flier, c in zip(ff['fliers'], newColor):
        flier.set(marker='o', color=c, alpha=0.5)
    plt.ylabel('AQI')
    # plt.savefig('result/SMAPE.png', bbox_inches='tight',dpi=600)  # 高清图
    # fig.subplots_adjust(right=0.82)  # 调整边距和子图的间距
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\箱线图.png', bbox_inches='tight', dpi=600)
    plt.show()

def jubufangda_MSO_8(mappingNumber, enhanceNumber):
    startPos = 270
    endIndex = 321
    MAX_EPISODES = endIndex - startPos
    x_axis_data = []
    for l in range(MAX_EPISODES):
        x_axis_data.append(l)

    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\ESN\\test\\ESN_resSize_920.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\GRU\\test\\GRU_MSO_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\LSTM\\test\\LSTM_MSO_predict.csv"

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1, 1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(-1, 1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(-1, 1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # plt.plot(x_axis_data, real_data, '-.1', color='#ef5b9c', label='real')
    # plt.plot(x_axis_data, BLS_data, '-.2', color='#f47920', label='BLS')
    # plt.plot(x_axis_data, ESN_data, '-.3', color='#b2d235', label='ESN')
    # plt.plot(x_axis_data, CFBLS_data, '-.4', color='#817936', label='CMBLS')
    # # plt.plot(x_axis_data, BLS_ESN_data, '-.>', color='#f15a22', label='BLS_ESN')
    # plt.plot(x_axis_data, CFBLS_ESN_data, '-.+', color='#009ad6', label='CMBESN')
    # plt.plot(x_axis_data, GRU_data, '-.*', color='#8552a1', label='GRU')
    # # plt.plot(x_axis_data, LSTM_data, '-.|', color='#ffc20e', label='LSTM')
    plt.plot(x_axis_data, real_data, color='#ef5b9c', label='real')
    plt.plot(x_axis_data, BLS_data,  color='#f47920', label='BLS')
    plt.plot(x_axis_data, ESN_data, color='#b2d235', label='ESN')
    plt.plot(x_axis_data, CFBLS_data,  color='#817936', label='CMBLS')
    # plt.plot(x_axis_data, BLS_ESN_data, '-.>', color='#f15a22', label='BLS_ESN')
    plt.plot(x_axis_data, CFBLS_ESN_data,  color='#009ad6', label='CMBESN')
    plt.plot(x_axis_data, GRU_data, color='#8552a1', label='GRU')
    # plt.plot(x_axis_data, LSTM_data, '-.|', color='#ffc20e', label='LSTM')

    plt.ylabel('MSO Value', fontsize = 14)
    plt.xlabel('Observed Point', fontsize = 14)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 把图例设置在外边
    plt.legend(loc='upper left', prop={'size' : 14})  # 把图例设置在外边
    # plt.ylabel('Death number')

    # 嵌入局部放大图的坐标系
    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.52, 0.62, 0.8, 1.2),
                       bbox_transform=ax.transAxes)
    # 在子坐标系中绘制原始数据

    axins.plot(x_axis_data, real_data, color='#ef5b9c', alpha=0.8, linewidth=1.5)
    axins.plot(x_axis_data, BLS_data,  color='#f47920', alpha=0.8, linewidth=1.5)
    axins.plot(x_axis_data, ESN_data,  color='#b2d235', alpha=0.8, linewidth=1.5)
    axins.plot(x_axis_data, CFBLS_data, color='#817936', alpha=0.8, linewidth=1.5)
    axins.plot(x_axis_data, BLS_ESN_data,  color='#f15a22', alpha=0.8, linewidth=1.5)
    axins.plot(x_axis_data, CFBLS_ESN_data,  color='#009ad6', alpha=0.8, linewidth=1.5)
    plt.plot(x_axis_data, GRU_data,  color='#8552a1', alpha=0.8, linewidth=1.5)
    plt.plot(x_axis_data, LSTM_data, color='#ffc20e', alpha=0.8, linewidth=1.5)

    # 设置放大区间，调整子坐标系的显示范围
    # 设置放大区间
    zone_left = 25
    zone_right = 29

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.0  # x轴显示范围的扩展比例
    y_ratio = 0.07  # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = x_axis_data[zone_left] - (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio
    xlim1 = x_axis_data[zone_right] + (x_axis_data[zone_right] - x_axis_data[zone_left]) * x_ratio

    # Y轴的显示范围
    y = np.hstack((real_data[zone_left:zone_right], BLS_data[zone_left:zone_right],
                   ESN_data[zone_left:zone_right], CFBLS_data[zone_left:zone_right],
                    CFBLS_ESN_data[zone_left:zone_right],GRU_data[zone_left:zone_right]))

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
    xy = (xlim0, ylim1)  # (60,-409)
    xy2 = (xlim0, ylim0)  # (60,409)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, color='#b22c46')
    axins.add_artist(con)

    xy = (xlim1, ylim1)  # (90,-409)
    xy2 = (xlim1, ylim0)  # (90,-409)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax, color='#b22c46')
    axins.add_artist(con)

    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\CM_局部放大图_折线图.svg', format='svg', bbox_inches='tight', dpi=800)  # 高清图
    # plt.show()

    x1 = np.arange(endIndex-startPos)
    fig1, vax = plt.subplots(1, 1, figsize=(11, 5))
    ERROR_ESN = []
    ERROR_BLS = []
    ERROR_CFBLS = []
    ERROR_CFBESN = []
    ERROR_GRU = []

    for i in range(endIndex-startPos):
        ERROR_ESN.append(abs(ESN_data[i] - real_data[i]))
        ERROR_BLS.append(abs(BLS_data[i] - real_data[i]))
        ERROR_CFBLS.append(abs(CFBLS_data[i] - real_data[i]))
        ERROR_CFBESN.append(abs(CFBLS_ESN_data[i] - real_data[i]))
        ERROR_GRU.append(abs(GRU_data[i] - real_data[i]))

    ERROR_ESN = np.array(ERROR_ESN).reshape(-1, 1)
    ERROR_BLS = np.array(ERROR_BLS).reshape(-1, 1)
    ERROR_CFBLS = np.array(ERROR_CFBLS).reshape(-1, 1)
    ERROR_CFBESN = np.array(ERROR_CFBESN).reshape(-1, 1)
    ERROR_GRU = np.array(ERROR_GRU).reshape(-1, 1)

    # plt.vlines(x1, [0], ERROR_ESN, colors='#00a6ac', label='ESN')
    # plt.vlines(x1, [0], ERROR_BLS, colors='#9b95c9', label='BLS')
    # plt.vlines(x1, [0], ERROR_CFBLS, colors='#fdb933', label='CMBLS')
    # plt.vlines(x1, [0], ERROR_CFBESN, colors='#a3cf62', label='GRU')
    # plt.vlines(x1, [0], ERROR_GRU, colors='#f05b72', label='CMBESN')

    # plt.plot(x, ERROR_ESN, color='#b2d235', label='ESN')
    # plt.plot(x, ERROR_BLS, color='#f47920', label='BLS')
    # plt.plot(x, ERROR_CFBLS, color='#817936', label='CFESN')
    # plt.plot(x, ERROR_CFBESN, color='#8552a1', label='CFBESN')
    # plt.plot(x, ERROR_GRU, color='#009ad6', label='GRU')

    plt.scatter(x1,  ERROR_ESN, s=18, marker='.', c='#00a6ac', label='ESN')
    plt.scatter(x1,  ERROR_BLS, s=18, marker='o', c='#9b95c9', label='BLS')
    plt.scatter(x1,  ERROR_CFBLS, s=18, marker='v', c='#fdb933', label='CMBLS')
    plt.scatter(x1,  ERROR_CFBESN, s=18, marker='p', c='#a3cf62', label='GRU')
    plt.scatter(x1,  ERROR_GRU, s=18, marker='D', c='#f05b72', label='CMBESN')
    plt.ylabel('Errors', fontsize = 14)
    plt.xlabel('Time steps', fontsize = 14)
    # fig1.subplots_adjust(right=0.88)  # 调整边距和子图的间距
    # plt.legend(loc='lower left')  # 把图例设置在外边
    # plt.legend(loc='upper left', bbox_to_anchor=(0.995, 1))  # 把图例设置在外边
    plt.legend(loc='upper right', prop={'size': 13})  # 把图例设置在外边
    # fig.subplots_adjust(right=1.5)  # 调整边距和子图的间距
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\CM_所有模型_误差垂线图_补审稿人1_加绝对值.svg', format='svg',bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_ERROR_MSO_8(mappingNumber, enhanceNumber):
    '''
    将所有模型的误差绘制在同一张图上
    startPos = 270
    endIndex = 321

    fig = plt.figure(figsize=(12, 5))
    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\ESN\\test\\ESN_resSize_920.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\GRU\\test\\GRU_MSO_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\LSTM\\test\\LSTM_MSO_predict.csv"

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1, 1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(-1, 1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(-1, 1)

    x1 = np.arange(endIndex - startPos)
    ERROR_ESN = []
    ERROR_BLS = []
    ERROR_CFBLS = []
    ERROR_CFBESN = []
    ERROR_GRU = []

    for i in range(endIndex - startPos):
        ERROR_ESN.append(abs(ESN_data[i] - real_data[i]))
        ERROR_BLS.append(abs(BLS_data[i] - real_data[i]))
        ERROR_CFBLS.append(abs(CFBLS_data[i] - real_data[i]))
        ERROR_CFBESN.append(abs(CFBLS_ESN_data[i] - real_data[i]))
        ERROR_GRU.append(abs(GRU_data[i] - real_data[i]))

    ERROR_ESN = np.array(ERROR_ESN).reshape(-1, 1)
    ERROR_BLS = np.array(ERROR_BLS).reshape(-1, 1)
    ERROR_CFBLS = np.array(ERROR_CFBLS).reshape(-1, 1)
    ERROR_CFBESN = np.array(ERROR_CFBESN).reshape(-1, 1)
    ERROR_GRU = np.array(ERROR_GRU).reshape(-1, 1)

    # plt.vlines(x1, [0], ERROR_ESN, colors='#00a6ac', label='ESN')
    # plt.vlines(x1, [0], ERROR_BLS, colors='#9b95c9', label='BLS')
    # plt.vlines(x1, [0], ERROR_CFBLS, colors='#fdb933', label='CMBLS')
    # plt.vlines(x1, [0], ERROR_CFBESN, colors='#a3cf62', label='GRU')
    # plt.vlines(x1, [0], ERROR_GRU, colors='#f05b72', label='CMBESN')

    # plt.plot(x, ERROR_ESN, color='#b2d235', label='ESN')
    # plt.plot(x, ERROR_BLS, color='#f47920', label='BLS')
    # plt.plot(x, ERROR_CFBLS, color='#817936', label='CFESN')
    # plt.plot(x, ERROR_CFBESN, color='#8552a1', label='CFBESN')
    # plt.plot(x, ERROR_GRU, color='#009ad6', label='GRU')

    plt.scatter(x1, ERROR_ESN, s=18, marker='.', c='#00a6ac', label='ESN')
    plt.scatter(x1, ERROR_BLS, s=18, marker='o', c='#9b95c9', label='BLS')
    plt.scatter(x1, ERROR_CFBLS, s=18, marker='v', c='#fdb933', label='CMBLS')
    plt.scatter(x1, ERROR_CFBESN, s=18, marker='p', c='#a3cf62', label='GRU')
    plt.scatter(x1, ERROR_GRU, s=18, marker='D', c='#f05b72', label='CMBESN')
    plt.ylabel('Errors', fontsize=14)
    plt.xlabel('Time steps', fontsize=14)
    # fig1.subplots_adjust(right=0.88)  # 调整边距和子图的间距
    # plt.legend(loc='lower left')  # 把图例设置在外边
    # plt.legend(loc='upper left', bbox_to_anchor=(0.995, 1))  # 把图例设置在外边
    plt.legend(loc='upper right', prop={'size': 13})  # 把图例设置在外边
    # fig.subplots_adjust(right=1.5)  # 调整边距和子图的间距
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\CM_所有模型_误差垂线图_补审稿人1_加绝对值.svg', format='svg',
    #             bbox_inches='tight', dpi=600)  # 高清图
    plt.show()
    '''

    startPos = 350
    endIndex = 400

    real_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\realData\\MSO_8\\test\\testReal.csv'
    ESN_path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\ESN\\test\\ESN_resSize_920.csv'
    BLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS\\test\\BLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS\\test\\CFBLS_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    BLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\BLS_ESN\\test\\BLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    CFBLS_ESN_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\CFBLS_ESN\\test\\CFBLS_ESN_mapping_{}_enhance_{}.csv".format(
        mappingNumber, enhanceNumber)
    GRU_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\GRU\\test\\GRU_MSO_predict.csv"
    LSTM_path = "E:\\yan_2\\CFBLS_LCFBLS复现\\result\\csvFile\\MSO_8\\LSTM\\test\\LSTM_MSO_predict.csv"

    real_data = pd.read_csv(real_path).values[startPos:endIndex].reshape(-1, 1)
    ESN_data = pd.read_csv(ESN_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_data = pd.read_csv(BLS_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_data = pd.read_csv(CFBLS_path).values[startPos:endIndex].reshape(-1, 1)
    BLS_ESN_data = pd.read_csv(BLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    CFBLS_ESN_data = pd.read_csv(CFBLS_ESN_path).values[startPos:endIndex].reshape(-1, 1)
    GRU_data = pd.read_csv(GRU_path).values[startPos:endIndex].reshape(-1, 1)
    LSTM_data = pd.read_csv(LSTM_path).values[startPos:endIndex].reshape(-1, 1)

    x1 = np.arange(endIndex - startPos)
    ERROR_ESN = []
    ERROR_BLS = []
    ERROR_CFBLS = []
    ERROR_CFBESN = []
    ERROR_GRU = []

    for i in range(endIndex - startPos):
        ERROR_ESN.append(abs(ESN_data[i] - real_data[i]))
        ERROR_BLS.append(abs(BLS_data[i] - real_data[i]))
        ERROR_CFBLS.append(abs(CFBLS_data[i] - real_data[i]))
        ERROR_CFBESN.append(abs(CFBLS_ESN_data[i] - real_data[i]))
        ERROR_GRU.append(abs(GRU_data[i] - real_data[i]))

    ERROR_ESN = np.array(ERROR_ESN).reshape(-1, 1)
    ERROR_BLS = np.array(ERROR_BLS).reshape(-1, 1)
    ERROR_CFBLS = np.array(ERROR_CFBLS).reshape(-1, 1)
    ERROR_CFBESN = np.array(ERROR_CFBESN).reshape(-1, 1)
    ERROR_GRU = np.array(ERROR_GRU).reshape(-1, 1)
    gs = gridspec.GridSpec(2, 6) #创建2行6列的网格

    lim_xy_max = max(max(ERROR_ESN), max(ERROR_BLS), max(ERROR_CFBLS), max(ERROR_CFBESN), max(ERROR_GRU)) + 0.1

    plt.figure(figsize=(12, 6))
    plt.subplot(gs[0, :2]) #gs[哪一行，列的范围]
    gs.update(hspace=0.3, wspace=0.8)

    plt.scatter(x1, ERROR_ESN, s=40, marker='.', c='#00a6ac', label='ESN')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 11})

    plt.subplot(gs[0, 2:4]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_BLS, s=40, marker='.', c='#9b95c9', label='BLS')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 11})

    plt.subplot(gs[0, 4:6]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_CFBLS, s=40, marker='.', c='#fdb933', label='CMBLS')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 11})

    plt.subplot(gs[1, 1:3]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_CFBESN, s=40, marker='.', c='#a3cf62', label='GRU')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 12})

    plt.subplot(gs[1, 3:5]) #gs[哪一行，列的范围]
    plt.scatter(x1, ERROR_GRU, s=40, marker='.', c='#f05b72', label='CMBESN')
    plt.ylim(0, lim_xy_max)
    plt.ylabel('Errors', fontsize=13)
    plt.xlabel('Time steps', fontsize=13)
    plt.legend(loc='upper right', prop={'size': 12})


    # plt.subplots_adjust(hspace=0.3,wspace=0.5)  # 调节上下子图之间的行距

    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\CM_所有模型_误差垂线图_补审稿人1_加绝对值.svg', format='svg',
                            bbox_inches='tight', dpi=600)  # 高清图
    plt.show()





def plot_3D_weipinwen_MSO_8():
    '''
    没有使用自平稳优化指标体系的3D绘图
    :return:
    '''
    data_path = r'E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\未使用自平稳指标体系的3D图.xlsx'
    data = pd.read_excel(data_path).values[:,:]
    # print(data)
    X = np.linspace(14,22,5)
    Y = np.linspace(2,28,14)
    z = np.array(data[:,3:4]).reshape(14,5)

    # print("x:",x)
    # print("y:", y)
    # print("z:", z)

    '''
    cmap = "以下的取值"
    'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 
    'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 
    'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 
    'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 
    'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 
    'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 
    'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 
    'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 
    'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
    'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 
    'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 
    'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 
    'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

    '''

    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(13, 8))
    ax = plt.axes(projection='3d')
    # ax.plot_wireframe(X, Y, z, rstride = 1, cstride = 1, cmap='RdPu')
    ax.plot_surface(X, Y, z, rstride=1, cstride=1, alpha=0.3,cmap="hsv_r", linewidth=0, antialiased=False)

    ax.contour(X, Y, z, zdir='x', offset=13, cmap="hsv_r")
    # ax.contour(X, Y, z, zdir='y', offset=30, cmap=cm.coolwarm)


    ax.set_xlabel('mapping node')
    ax.set_ylabel('ESN')
    ax.set_zlabel('RMSE', rotation=90)
    # ax.grid(False, linestyle = "-.", color = "red", linewidth = "1")

    ax.view_init(elev=18, azim=45)


    # ax.contour(X, Y, z, cmap=cm.coolwarm)
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\未使用自平稳指标体系的3D图.png', bbox_inches='tight', dpi=600)

    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    plt.show()

def plot_3D_pinwen_MSO_8():
    '''
    没有使用自平稳优化指标体系的3D绘图
    :return:
    '''
    data_path = r'E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\自平稳指标体系的3D图.xlsx'
    data = pd.read_excel(data_path).values[:,:]
    # print(data)
    X = np.linspace(14,36,12)
    Y = np.linspace(2,28,14)
    z = np.array(data[:,3:4]).reshape(14,12)

    X, Y = np.meshgrid(X, Y)
    fig = plt.figure(figsize=(13, 8))
    ax = plt.axes(projection='3d')
    # ax.plot_wireframe(X, Y, z, rstride = 1, cstride = 1, cmap='RdPu')
    ax.plot_surface(X, Y, z, rstride=2, cstride=2, alpha=0.3,cmap="hsv", linewidth=0, antialiased=False)

    ax.contour(X, Y, z, zdir='x', offset=13, cmap="hsv_r")
    # ax.contour(X, Y, z, zdir='z', offset=0, cmap="hsv_r")


    ax.set_xlabel('mapping node')
    ax.set_ylabel('ESN')
    ax.set_zlabel('RMSE', rotation=90)
    # ax.grid(False, linestyle = "-.", color = "red", linewidth = "1")

    ax.view_init(elev=11, azim=49)


    # ax.contour(X, Y, z, cmap=cm.coolwarm)
    # plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\使用自平稳指标体系的3D图.png', bbox_inches='tight', dpi=600)

    # 调整观察角度和方位角。这里将俯仰角设为60度，把方位角调整为35度
    plt.show()

def plot_real_data_fangshan():
    path = 'E:\\yan_1\\BLS_self\\fangshan.csv'
    AQI = pd.read_csv(path).values[1000:2000, :1].reshape(-1, 1)
    # AQI = pd.read_csv(path).values[1000:15000, :1].reshape(-1, 1)

    x = np.arange(AQI.shape[0])
    plt.figure(figsize=(13, 6))


    plt.plot(x, AQI, color='#6950a1',label = "real data", linewidth=2)
    plt.ylabel('AQI',fontsize = 14)
    plt.xlabel('Time steps',fontsize = 14)
    plt.legend(loc='upper right', prop={'size':13})  # 把图例设置在外边
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\第一版修改的图\\真实数据图.svg', format='svg',bbox_inches='tight', dpi=600)  # 高清图
    plt.show()


def plot_real_data_MSO_8():
    path = 'E:\\yan_2\\CFBLS_LCFBLS复现\\dataset\\MSO_data_8.csv'
    # data = pd.read_csv(path).values[1000:15000, :1].reshape(-1, 1)
    data = pd.read_csv(path).values[1000:2000, :1].reshape(-1, 1)
    x = np.arange(data.shape[0])
    plt.figure(figsize=(13, 6))

    plt.plot(x, data, color='#6950a1', label = "real data", linewidth=2)
    plt.ylabel('MSO Value', fontsize = 14)
    plt.xlabel('Observed Point', fontsize = 14)
    plt.legend(loc='upper right', prop={'size':13})  # 把图例设置在外边
    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\真实数据图.svg', format='svg',bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_zhuzhuangtu_2Y_MSO():
    # plt.rcParams['axes.labelsize'] = 16  # x轴ticks的size
    plt.rcParams['xtick.labelsize'] = 13  # x轴ticks的size
    ESN = np.array([33.5238, 36.5234, 36.1198,0])
    BLS = np.array([5.4200, 0.3067, 16.9903, 0])
    GRU = np.array([3.0556, 9.722, 0.9779, 0])
    CMBLS = np.array([8.3990, 2.9850, 4.9296, 0])
    tempR2 = np.array([0.07, 0.0112, 0.0041, 0.0122])
    color = ['#d71345', '#843900', '#bed742', '#121a2a']

    a = np.random.random(5)
    print(a.shape)
    print(ESN.shape)
    x = np.arange(4)
    xLabel = ["SMAPE", "MAE", "RMSE", "R2"]
    total_width, n = 0.8, 4
    width = total_width / n

    fig, ax1 = plt.subplots(figsize=(11, 5))
    print(x)
    x2 = [2.7+width, 2.7+width*2, 2.7+width*3, 2.7+width*4]



    ax1.bar(x, ESN, width=width, color= '#d71345',label="ESN")
    ax1.bar(x + width, BLS, width=width, color= '#843900', label="BLS", tick_label=xLabel)
    ax1.bar(x + width*2, GRU, width=width, color= '#bed742', label="GRU")
    ax1.bar(x + width*3, CMBLS, width=width, color= '#121a2a', label="CMBLS")

    ax1.set_ylabel('percentage', fontsize = 14)
    ax1.yaxis.set_ticks_position('both')
    plt.legend(loc='upper right', prop={'size': 13})

    ax1.set_xticklabels(ax1.get_xticklabels())

    ax2 = ax1.twinx()
    ax2.set_ylabel('percentage', fontsize = 14)
    ax2.set_ylim(0, 0.1)
    ax2.bar(x2, tempR2, width=width, color=color, align='edge')

    plt.tight_layout()

    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\第一版修改的图\\table6后的图.svg', format='svg',bbox_inches='tight', dpi=600)  # 高清图
    plt.show()

def plot_zhuzhuangtu_2Y_fangshan():
    # plt.rcParams['axes.labelsize'] = 16  # x轴ticks的size
    plt.rcParams['xtick.labelsize'] = 13  # x轴ticks的size
    ESN = np.array([43.9564, 61.4283, 66.7504,90.1663])
    BLS = np.array([53.0653, 86.7112, 78.1442, 47.2040])
    GRU = np.array([45.9903, 11.5430, 48.1928, 32.3362])
    CMBLS = np.array([24.1191, 20.0722, 45.3040, 45.3040])

    color = ['#00a6ac' '#9b95c9' '#fdb933' '#a3cf62']

    x = np.arange(4)
    xLabel = ["SMAPE", "MAE", "RMSE", "R2"]
    total_width, n = 0.8, 4
    width = total_width / n

    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.set_ylim(0, 120)
    ax1.bar(x, ESN, width=width, color= '#00a6ac',label="ESN")
    ax1.bar(x + width, BLS, width=width, color= '#9b95c9', label="BLS", tick_label=xLabel)
    ax1.bar(x + width*2, GRU, width=width, color= '#fdb933', label="GRU")
    ax1.bar(x + width*3, CMBLS, width=width, color= '#a3cf62', label="CMBLS")

    ax1.set_ylabel('percentage', fontsize = 14)
    ax1.yaxis.set_ticks_position('both')
    plt.legend(loc='upper right', prop={'size': 13})

    plt.savefig('E:\\yan_2\\12_5论文\\实验结果图\\fangshan\\第一版修改的图\\table7后的图.svg', format='svg',bbox_inches='tight', dpi=600)  # 高清图
    plt.show()



if __name__ == '__main__':
    # zhexiantu_fangshan(32, 30)
    # fangshan_boxplot(32, 30)
    # getStationaryFigure()

    # zhexiantu_MSO_8(36, 24)
    # MSO_8_boxplot(36, 24)
    # jubufangda_MSO_8(36, 24)
    # plot_3D_weipinwen_MSO_8()
    # plot_3D_pinwen_MSO_8()
    # plot_3D_pinwen_fangshan()
    # plot_real_data_fangshan()
    # zhexiantu_fangshan(32, 30)
    # fangshan_boxplot(32, 30)
    # jubufangda_MSO_8(36, 24)
    # plot_real_data_MSO_8()
    # jubufangda_MSO_8(36, 28)

    '''
    低一次修改稿的修改内容
    '''
    # plot_real_data_fangshan()
    # plot_real_data_MSO_8()plot_zhuzhuangtu_2Y
    # jubufangda_MSO_8(36, 24)
    # plot_ERROR_MSO_8(36, 24)
    # plot_ERROR_fangshan(32, 30)
    # zhexiantu_fangshan(32, 30)
    # plot_zhuzhuangtu_2Y_MSO()
    plot_zhuzhuangtu_2Y_fangshan()