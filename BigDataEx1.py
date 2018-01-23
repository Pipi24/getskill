import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier




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


def init(fcate_8, fcate8, X, item):
    for index in range(len(fcate_8)):
        i = fcate_8[index]
        print(i, item[i])
        print(X[:, i])
        fcate8(X[:, i])
        print(X[:, i])


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


def getIv(d, total):
    IV = {}
    sum = 0;
    for index, value in d.items():
        if total[1] == 0 or total[0] == 0 or value[0] == 0 or value[1] == 0:
            continue
        temp = (value[1] / total[1]) / (value[0] / total[0])
        woe = np.log(temp)
        iv = ((value[1] / total[1]) - (value[0] / total[0])) * (woe)
        # iv = (value[1] - value[0]) * (woe)
        IV[index] = iv
    for index, value in IV.items():
        sum = sum + value
    return IV, sum


data = pd.read_csv('data.csv', encoding='GBK')
m = np.array(data)


errorData = []

for index in range(len(m)):
    item = m[index]
    for num in item:
        if num != num:
            errorData.append(index)
M = np.delete(m, errorData, 0)


X = M[:, 0:32]
y = M[:, -1]
item = {}
i = 0
for key in data.keys():
    item[i] = key
    print(i,": ", key)
    i = i + 1


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


print("IV")
totalIV = []
for index in range(len(X[0])):
    d,total = getNum(X[:, index])
    IV,sum = getIv(d,total)
    if sum >= 0.03:
        print(index, item[index], sum)
    totalIV.append(sum)


str = []
for k,v in item.items():
    str.append(v)
file = {}
for index in range(len(X[0])):
    file[str[index]] = X[:, index]
file['Y_FLAG'] = y
f = pd.DataFrame(file)
f.to_csv('output.csv',index=False, columns=str)


def work():

    data = pd.read_csv('output.csv', encoding='GBK')
    m = np.array(data)
    X = m[:, 0:32]
    y = m[:, -1]


    testSum = 20000
    xtrain = X[0:testSum]
    ytrain = y[0:testSum]
    length = len(X)
    xtest = X[testSum:length - 1]
    ytest = y[testSum:length - 1]


    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)
    print(model)

    expected = ytest
    predicted = model.predict(xtest)

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print("score: ", metrics.accuracy_score(y_true=expected, y_pred=predicted))

work()
