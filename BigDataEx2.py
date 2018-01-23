from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from io import StringIO

import numpy as np
import pandas as pd
import csv

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# 以下以cate开头均为对变量重新分档

def cate8(array):
    for index, value in enumerate(array):
        if value < 12.5:
            array[index] = 0
        elif value < 25:
            array[index] = 12.5
        elif value < 37.5:
            array[index] = 25
        elif value < 50:
            array[index] = 37.5
        elif value < 62.5:
            array[index] = 50
        elif value < 75:
            array[index] = 62.5
        elif value < 87.5:
            array[index] = 75
        elif value >= 87.5:
            array[index] = 87.5

def cate6(array):
    for index, value in enumerate(array):
        if value == 0:
            array[index] = 0
        elif value < 20:
            array[index] = 0
        elif value < 40:
            array[index] = 20
        elif value < 60:
            array[index] = 40
        elif value < 80:
            array[index] = 60
        elif value >= 80:
            array[index] = 80

def cate4(array):
    for index, value in enumerate(array):
        if value < 25:
            array[index] = 0
        elif value < 50:
            array[index] = 25
        elif value < 75:
            array[index] = 50
        elif value >= 75:
            array[index] = 75

def cate3(array):
    for index, value in enumerate(array):
        if value < 33.3:
            array[index] = 0
        elif value < 66.6:
            array[index] = 33.3
        if value >= 66.6:
            array[index] = 66.6

def cate3zero(array):
    for index, value in enumerate(array):
        if value == 0:
            array[index] = 0
        elif value < 50:
            array[index] = 0
        if value >= 50:
            array[index] = 50

def cate2(array):
    for index, value in enumerate(array):
        if value == 'high':
            array[index] = 1
        if value == 'low':
            array[index] = 0

def cate2zero(array):
    for index, value in enumerate(array):
        if value == 0:
            array[index] = 0
        if value != 0:
            array[index] = 1

# 数据初始化，即组织数据分档
def init(fcate_8, fcate8, X, item):
    for index in range(len(fcate_8)):
        i = fcate_8[index]
        print(i, item[i])
        print(X[:, i])
        fcate8(X[:, i])
        print(X[:, i])

# 获取对应的高费用和低费用总数

def getNum(re):
    d = {}
    for i in range(len(re)):
        temp = re[i]
        value = y[i]

        if d.get(temp, -1) == -1:
            d[temp] = []
            d[temp].append(0)
            d[temp].append(0)

        if value == 1:
            d[temp][1] = d[temp][1] + 1
        else:
            d[temp][0] = d[temp][0] + 1

    total = [0,0]
    for index, value in d.items():
        total[0] = value[0] + total[0]
        total[1] = value[1] + total[1]

    return d,total

# 计算总的IV值
def getIv(d, total):
    IV = {}
    sum = 0;
    for index,value in d.items():
        if total[1] == 0 or total[0] == 0 or value[0] == 0 or value[1] == 0:
            continue
        temp = (value[1]/total[1])/(value[0]/total[0])
        woe = np.log(temp)
        iv = ((value[1]/total[1]) - (value[0]/total[0]))*(woe)
        # iv = (value[1] - value[0]) * (woe)
        IV[index] = iv
    for index,value in IV.items():
        sum = sum + value
    return IV, sum

def loadRecords(fileNameContents):
    """读取给定文件中的所有记录"""
    input = StringIO(fileNameContents[1])
    reader = csv.DictReader(input, fieldnames=["name", "favoriteAnimal"])
    return reader

# 引入PySpark，spark初始化配置
conf = SparkConf().setMaster("local[*]").setAppName("Fisrt")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# 读入数据
inputFile = 'data.csv'
fullFileData = sc.wholeTextFiles(inputFile).flatMap(loadRecords)
headers = fullFileData.first()
# 获取表头
name = []
for k,v in headers.items():
    if k == None:
        for index in range(len(v)):
            name.append(v[index])
    else:
        name.append(v)

# 获取具体数据内容
context = fullFileData.collect()
context.pop(0)
data = []
# 将获得的数据转化为多维数组
for index in range(len(context)):
    temp = []
    item = context[index]
    for k,v in item.items():
        if k == None:
            for index in range(len(v)):
                temp.append(v[index])
        else:
            temp.append(v)
    data.append(temp)

# 将多维数组转为float的多维数据，对于提出含空属性的数据，最后一列费用高低属性不变
dataum = []

for index in range(len(data)):
    tmp = data[index]
    num = []
    a = True
    for i in range(len(tmp)):
        if tmp[i] == '':
            a = False
            break;
        if i == len(tmp) - 1:
            num.append(tmp[i])
        else:
            num.append(float(tmp[i]))

    if a:
        dataum.append(num)

# 将数据转为np.array方便后期处理
df = pd.DataFrame(dataum)
m = np.array(df)

# 切片，分离属性和标签
X = m[:,0:32]
y = m[:, -1]
item = {}
for index in range(len(name)):
    item[index] = name[index]
    print(index,": ", name[index])


# 进行数据重新分档
cate_8 = [1,7,8,9,11,22,24]
cate_6 = [2,3,4,5,10,12,13,26]
cate_4 = [6]
cate_3 = [20, 21, 23, 25, 27, 28, 29]
cate_3_zero = [17, 19]
cate_2 = [32]
cate_2_zero = [14, 15, 16, 18]

init(cate_8, cate8, X, item)
init(cate_6, cate6, X, item)
init(cate_4, cate4, X, item)
init(cate_3, cate3, X, item)
init(cate_3_zero, cate3zero, X, item)
cate2(y)
init(cate_2_zero, cate2zero, X, item)

# 计算IV值
print("IV")
totalIV = []
for index in range(len(X[0])):
    d,total = getNum(X[:, index])
    IV,sum = getIv(d,total)
    if sum >= 0.03:
        print(index, item[index], sum)
    totalIV.append(sum)

# 写入csv文件
str = []
for k,v in item.items():
    str.append(v)
file = {}
for index in range(len(X[0])):
    file[str[index]] = X[:, index]
file['Y_FLAG'] = y
f = pd.DataFrame(file)
f.to_csv('output.csv',index=False, columns=str)


# 进行机器学习
def work():
    # 读入文件
    data = pd.read_csv('output.csv', encoding='GBK')
    m = np.array(data)
    X = m[:, 0:32]
    y = m[:, -1]

    # 确定测试数据集和训练数据集
    testSum = 19000
    xtrain = X[0:testSum]
    ytrain = y[0:testSum]
    length = len(X)
    xtest = X[testSum:length - 1]
    ytest = y[testSum:length - 1]

    # 决策树训练
    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)
    print(model)
    # 检验分类器
    # make predictions
    expected = ytest
    predicted = model.predict(xtest)

    # 输出分类器的相关参数和准确率
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("score: ", metrics.accuracy_score(y_true=expected, y_pred=predicted))

work()



