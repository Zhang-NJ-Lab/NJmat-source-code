import matplotlib.pyplot as plot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
#_random_over_sampler
import pickle

data = pd.DataFrame(pd.read_csv('data_rfe.csv'))

X = data.values[:, :-1]
y = data.values[:, -1]


ros = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=1)

for i in range(X_train.shape[1]):
    X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

for i in range(X_test.shape[1]):
    X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])



clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
Classified_two_RF = clf.fit(X_train, y_train)

# 画出ROC曲线 RandomForest test
y_score = Classified_two_RF.predict_proba(X_test)
fpr, tpr, threshold = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
print(fpr)
print(tpr)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.01, 1.0])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title('AUC')
plt.legend(loc="lower right", fontsize=20, frameon=False)
# plt.show()
plt.savefig('RandomForest_test_ROC.png', dpi=300, bbox_inches='tight')
plt.close()

# 画出混淆矩阵 RandomForest test
# clf.fit(X, y)
prey = Classified_two_RF.predict(X_test)
true = 0
for i in range(0, len(y_test)):
    if prey[i] == y_test[i]:
        true = true + 1
C = confusion_matrix(y_test, prey, labels=[0, 1])
plt.imshow(C, cmap=plt.cm.Blues)
indices = range(len(C))
plt.xticks(indices, [0, 1], fontsize=20)
plt.yticks(indices, [0, 1], fontsize=20)
plt.colorbar()
for first_index in range(len(C)):  # 第几行
    for second_index in range(len(C)):  # 第几列
        plt.text(first_index, second_index, C[first_index][second_index], fontsize=20, horizontalalignment='center')
# plt.show()
plt.savefig('/RandomForest_test_CM.png', dpi=300, bbox_inches='tight')
plt.close()
print("true:", true)
str2 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

# 画出ROC曲线 RandomForest train的AUC
y_score = Classified_two_RF.predict_proba(X_train)
fpr, tpr, threshold = roc_curve(y_train, y_score[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
print(fpr)
print(tpr)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.01, 1.0])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('AUC')
plt.legend(loc="lower right", fontsize=20, frameon=False)
# plt.show()
plt.savefig('RandomForest_train_ROC.png', dpi=300, bbox_inches='tight')
plt.close()

# 画出混淆矩阵 RandomForest train 混淆矩阵
prey = Classified_two_RF.predict(X_train)
true = 0
for i in range(0, len(y_train)):
    if prey[i] == y_train[i]:
        true = true + 1
C = confusion_matrix(y_train, prey, labels=[0, 1])
plt.imshow(C, cmap=plt.cm.Blues)
indices = range(len(C))
plt.xticks(indices, [0, 1], fontsize=20)
plt.yticks(indices, [0, 1], fontsize=20)
plt.colorbar()
for first_index in range(len(C)):  # 第几行
    for second_index in range(len(C)):  # 第几列
        plt.text(first_index, second_index, C[first_index][second_index], fontsize=20, horizontalalignment='center')
# plt.show()
plt.savefig('/RandomForest_train_CM.png', dpi=300, bbox_inches='tight')
plt.close()
print("true:", true)
str3 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

pickle.dump(Classified_two_RF, open("/Classified_two_RF.dat", "wb"))

