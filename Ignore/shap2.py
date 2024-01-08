import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import numpy as np
# 将 np.bool 替换为 np.bool_
np.bool = np.bool_

print(1)


import joblib# 加载训练好的ExtraTreeClassifier模型
model = joblib.load(open("Continuous_RF.dat","rb"))

print(1)

data = pd.read_csv('data_rfe.csv')
# data = data.iloc[:,1:]
# data

print(1)

# 拟合模型
X = data.drop(columns=['Potential (v)'])
y = data['Potential (v)']

print(1)

# 初始化 SHAP explainer
explainer = shap.Explainer(model, X)

# 计算 SHAP 值
shap_values = explainer(X)

print(1)

# 将 shap 值转换为 pandas DataFrame
shap_df = pd.DataFrame(shap_values.values[:,:,1], columns=X.columns)



