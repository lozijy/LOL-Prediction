import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import csv
data = pd.read_csv("../data/high_diamond_ranked_10min.csv")
data.head
data.describe()
data.shape

#查看相关性绘制热力图

sns.set(context='notebook', style='ticks', font_scale=0.5)
corr = data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

#预处理
data.drop(["gameId"
           ,"blueWardsPlaced"
           ,"blueWardsDestroyed"
           #,"blueAssists"
           #,"blueEliteMonsters"
           ,"blueAvgLevel"
           ,"blueTotalJungleMinionsKilled"
           ,"blueCSPerMin"
           ,"blueGoldPerMin"
           ,"blueGoldDiff"
           ,"blueExperienceDiff"
           ,"blueTotalExperience"
           ,"redWardsPlaced"
           ,"redWardsDestroyed"
           ,"redFirstBlood"
           ,"redKills"
           ,"redDeaths"
           #,"redAssists"
           #,"redEliteMonsters"
           ,"redAvgLevel"
           ,"redTotalJungleMinionsKilled"
           ,"redGoldDiff"
           ,"redExperienceDiff"
           ,"redTotalExperience"
           ,"redCSPerMin"
           ,"redGoldPerMin"],inplace = True ,axis = 1)
labels=np.array(data["blueWins"])
features=data.drop("blueWins",axis=1)
feature_list=list(features.columns)
features=np.array(features)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.15, random_state=123)
print(X_train.shape)

dtree = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3, # 定义树的深度, 可以用来防止过拟合
    min_weight_fraction_leaf=0.01 # 定义叶子节点最少需要包含多少个样本(使用百分比表达), 防止过拟合
    )
dtree.fit(X_train,y_train)
dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))
print ("决策树 AUC = %2.2f" % dt_roc_auc)
print(classification_report(y_test, dtree.predict(X_test)))


export_graphviz(dtree,out_file="../aboutIMG/tree.dot",feature_names=feature_list,rounded=True,precision=1)
(graph,) = pydot.graph_from_dot_file("../aboutIMG/tree.dot")
graph.write_png("../aboutIMG/tree.png")

#看看效果
true_data=pd.DataFrame(data={"id":[i for i in range(100)],"actual":y_test[:100]})
plt.plot(true_data["id"],y_test[:100],"bp",label="actual")
plt.plot(true_data["id"],dtree.predict(X_test)[:100],"rp",label="prediction")

plt.legend()
plt.xlabel("id")
plt.ylabel("blueWin")
plt.title("Actual and Predicted Values")
plt.show()

#用户输入
lis = ['blueFirstBlood'
                 ,'blueKills'
                 ,'blueDeaths'
                 ,'blueAssists'
                 ,'blueEliteMonsters'
                 ,'blueDragons'
                 ,'blueHeralds'
                 ,'blueTowersDestroyed'
                 ,'blueTotalGold'
                 ,'blueTotalMinionsKilled'
                 ,'redAssists'
                 ,'redEliteMonsters'
                 ,'redDragons'
                 ,'redHeralds'
                 ,'redTowersDestroyed'
                 ,'redTotalGold'
                 ,'redTotalMinionsKilled']
lol_data = []
for i in range(17):
    print("Please Iuput " + lis[i] + " : ", end='')
    s = int(input())
    lol_data.append([s])

lol_data = np.array(lol_data)
header = np.array(lis)
with open('lol.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(header)
    f_csv.writerows(lol_data.T)

want_data = pd.read_csv(r'lol.csv')

print()
if dtree.predict(want_data) == 1:
    print("BlueTeamWin, Probability: " + str(dtree.predict_proba(want_data)[0][1]))
else:
    print("BlueTeamLose, Probability: " + str(dtree.predict_proba(want_data)[0][0]))

import os

os.remove('lol.csv')
