# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt     
plt.rcParams['font.sans-serif'] = ['SimHei']    #定义使其正常显示中文字体黑体
plt.rcParams['axes.unicode_minus'] = False      #用来正常显示表示负号
import warnings
warnings.filterwarnings("ignore")


data = pd.read_excel('brand_dazong.xlsx', index_col = u'日期',header = 0)
print(data.head())

#画出时序图
data.plot()
plt.show()

#画出自相关性图
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(data)
plt.show()

#平稳性检测
from statsmodels.tsa.stattools import adfuller
#返回值依次为：adf, pvalue p值， usedlag, nobs, critical values临界值 , 
# icbest, regresults, resstore 
#adf 分别大于3中不同检验水平的3个临界值，单位检测统计量对应的p 值显著大于 0.05 ， 
#说明序列可以判定为 非平稳序列
print('原始序列的检验结果为：',adfuller(data[u'销量']))

#对数据进行差分后得到 自相关图和 偏相关图
D_data = data.diff().dropna()
D_data.columns = [u'销量差分']

D_data.plot()   #画出差分后的时序图
plt.show()

plot_acf(D_data)    #画出自相关图
plt.show()
plot_pacf(D_data)   #画出偏相关图
plt.show()
#一阶差分后的序列的时序图在均值附近比较平稳的波动， 自相关性有很强的短期相关性， 
#单位根检验 p值小于 0.05 ，所以说一阶差分后的序列是平稳序列
print(u'差分序列的ADF 检验结果为： ', adfuller(D_data[u'销量差分']))   #平稳性检验

#对一阶差分后的序列做白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果：',acorr_ljungbox(D_data, lags= 1)) #返回统计量和 p 值
# 差分序列的白噪声检验结果： (array([*]), array([*])) p值为第二项， 远小于 0.05

#对模型进行定阶
from statsmodels.tsa.arima_model import ARIMA 

pmax = int(len(D_data) / 10)    #一般阶数不超过 length /10
qmax = int(len(D_data) / 10)
bic_matrix = []
for p in range(pmax +1):
    temp = []
    for q in range(qmax+1):
        try:
            value = ARIMA(D_data, (p, 1, q)).fit().bic
            temp.append(value)
        except:
            temp.append(None)
        bic_matrix.append(temp)

bic_matrix = pd.DataFrame(bic_matrix)   #将其转换成Dataframe 数据结构
p,q = bic_matrix.stack().idxmin()   #先使用stack 展平， 然后使用 idxmin 找出最小值的位置

print(u'BIC 最小的p值 和 q 值：%s,%s' %(p,q))  #  BIC 最小的p值 和 q 值：0,1
#所以可以建立ARIMA 模型
model = ARIMA(data, (p,1,q)).fit()
model.summary2()
#保存模型
model.save('model.pkl')
#模型加载
from statsmodels.tsa.arima_model import ARIMAResults
loaded = ARIMAResults.load('model.pkl')
#预测未来五个单位
predictions=loaded.forecast(5) 
#预测结果为：
pre_result = predictions[0]
print(u'预测结果为：',pre_result)
#标准误差为：
error = predictions[1]
print(u'标准误差为：',error)
#置信区间为：
confidence = predictions[2]
print(u'置信区间为：',confidence)     
  
     
     
