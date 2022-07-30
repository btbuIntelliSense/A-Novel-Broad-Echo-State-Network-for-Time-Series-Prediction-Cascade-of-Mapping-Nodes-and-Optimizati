from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd

path = 'E:\\yan_2\\12_5论文\\实验结果图\\MSO_8\\ADF检验数据.csv'
data = pd.read_csv(path).values[1:185,:]
data = np.array(data)
dftest = adfuller(data, regression='ctt', autolag='AIC')
'''
输入参数 :
regression：{‘c’，‘ct’，‘ctt’，‘nc’}， 包含在回归中的常量和趋势顺序。‘c’：仅限常量（默认值）。 ‘ct’：恒定和趋势。 ‘ctt’：常数，线性和二次趋势。 ‘nc’：没有恒定，没有趋势

返回值 :
ADF：float，测试统计。
pvalue：float，probability value：MacKinnon基于MacKinnon的近似p值（1994年，2010年）。
usedlag：int，使用的滞后数量。
NOBS：int，用于ADF回归和计算临界值的观察数。
critical values：dict，测试统计数据的临界值为1％，5％和10％。基于MacKinnon（2010）。
icbest：float，如果autolag不是None，则最大化信息标准。
resstore：ResultStore, optional，一个虚拟类，其结果作为属性附加。

'''
print(dftest)
print("%.50f"%dftest[1])
print("%.50f"%dftest[4]['1%'])
