# 1.封装函数file_name打开的文件名
# data是csv倒入时的数据集¶
# data_rfe在后面会有，是rfe特征选择后的总数据集
# s_rfe 是rfe特征选择后的特征数据
# target是目标数据

#1.1打开csv并存到data中
def file_name(name,path):
    import pandas as pd
    global data
    data = pd.read_csv(name)
    data.to_csv(path+"/data.csv")
    print(data)
    return data

#1.2画所有列分布的柱状图
def hist(path):
    import matplotlib.pyplot as plt
    # 绘制柱状图，其中bins设置为50
    data.hist(bins=50, figsize=(20,15))
    plt.tight_layout()
    plt.savefig(path+'/hist_allFeatures.png', dpi=300, bbox_inches = 'tight')
    plt.close()



#2.封装函数特征选择之前heatmap画热图
def heatmap_before(path):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    featureData=data.iloc[:,:]
    global corMat
    corMat = pd.DataFrame(featureData.corr())  #corr 求相关系数矩阵
    corMat.to_csv(path+'/heatmap-before.csv')
    plt.figure(figsize=(20, 30))
    sns.heatmap(corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
    plt.savefig(path+'/heatmap-before.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return


#3. rfe特征选择 feature_rfe_select1 is easier
def feature_rfe_select1(remain_number,path):
    # 使用随机森林的rfe:RandomForestRegressor()
    from sklearn import preprocessing
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.ensemble import RandomForestRegressor
    import csv
    import numpy as np

    # 输入数据归一化
    X = data.values[:, :-1]
    for i in range(X.shape[1]):
        X[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X[:, [i]])
    y = data.values[:, -1]  # 目标数值

    # rfe步骤
    model = RandomForestRegressor()
    rfe = RFE(estimator=model, n_features_to_select=remain_number, step=1)
    rfe_X = rfe.fit_transform(X, y)
    print("特征是否被选中：\n", rfe.support_)                                          # ndarray
    print("获取的数据特征尺寸:", rfe_X.shape)                                           # tuple
    list1 = rfe.support_.tolist()

    # 打印rfe后的特征，但可能包含空值
    import pandas as pd
    Features_0 = pd.DataFrame(data=data.iloc[:, :-1].columns, columns=['Features'])
    Features_0
    Features_rfe = pd.DataFrame(data=rfe.support_, columns=['whether selected'])
    Features_rfe
    #     pd.options.display.max_rows=None
    p = pd.concat([Features_0, Features_rfe], axis=1)
    q = p[p['whether selected']>0]
    r = q.reset_index(drop=True)
    global s_rfe
    s_rfe = pd.DataFrame(data=data,columns=r.Features.values)
    global target
    target = pd.DataFrame(data=data.iloc[:,-1])
    # target = pd.DataFrame(data, columns=['Potential (v)'])
    global data_rfe
    data_rfe = pd.concat([s_rfe,target], axis=1)
    print("最后的特征s_rfe:", r.Features.values)                                        # ndarray
    print("目标target:", target)
    print("rfe后的总数据data_rfe:", data_rfe)

    list2 = r.Features.values.tolist()

    # print全输出
    with open(path + "/data.txt", "w") as f:
        #f.write("特征是否被选中：\n")
        f.write("Whether the feature is selected:\n")
        for i in range(len(list1)):
            f.write(str(list1[i])+' ')
        #f.write("\n获取的数据特征尺寸：\n")
        f.write("\nAcquired data feature size:\n")
        f.write('(%s,%s)' % rfe_X.shape)
        #f.write("\n最后的特征s_rfe：\n")
        f.write("\nS_rfe(Final feature)：\n")
        for i in range(len(list2)):
            f.write(str(list2[i]) + '\n')
    target.to_csv(path + "/target.csv")
    data_rfe.to_csv(path + "/data_rfe.csv")
    return target,data_rfe

#4.1 画rfe特征选择后的热图
def heatmap_afterRFE(path):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    data_rfe_corMat = pd.DataFrame(data_rfe.corr())  #corr 求相关系数矩阵
    data_rfe_corMat.to_csv(path+'/heatmap-afterRFE.csv')
    plt.figure(figsize=(20, 30))
    sns.heatmap(data_rfe_corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
    plt.savefig(path+'/heatmap-afterRFE.png', dpi=300, bbox_inches = 'tight')
    plt.close()


#4.2 画rfe特征选择后的pairplot图
def pairplot_afterRFE(path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    g6 = sns.pairplot(data_rfe, kind='reg')
    plt.savefig(path+'/sns-pairplot-remain.png', dpi=300, bbox_inches = 'tight')
    plt.close()


#5 重要性排名（皮尔逊系数）
#5.1 特征选择之前所有特征的重要性
def FeatureImportance_before(rotationDeg,fontsize_axis,figure_size_xaxis,figure_size_yaxis,path):

    import pandas as pd
    FirstLine=corMat.iloc[-1,:]
    FirstLine=pd.DataFrame(FirstLine)
    FirstLine_Del_Target=FirstLine.iloc[:-1,:]
    importance=FirstLine_Del_Target.sort_values(by=FirstLine_Del_Target.columns.tolist()[-1],ascending=False)
    # importance=FirstLine_Del_Target.sort_values(by="Potential (v)",ascending=False)
    try:
        print(importance)
    except Exception as e:
        print(e)
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Times New Roman']
    # plt.rcParams ['font.sans-serif'] ='SimHei'    #显示中文
    plt.rcParams ['axes.unicode_minus']=False    #显示负号
    importance.plot(kind='bar', figsize=(figure_size_xaxis,figure_size_yaxis), rot=rotationDeg, fontsize=8)  #colormap='rainbow'

    plt.savefig(path+'/FeatureImportance_before.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return importance

#5.2 特征选择之后的个别特征的重要性
def FeatureImportance_afterRFE(rotationDeg, fontsize_axis, figure_size_xaxis, figure_size_yaxis,path):
    import pandas as pd
    corMat_rfe = pd.DataFrame(data_rfe.corr())  # corr 求相关系数矩阵

    FirstLine_rfe = corMat_rfe.iloc[-1, :]
    FirstLine_rfe = pd.DataFrame(FirstLine_rfe)
    FirstLine_rfe_Del_Target = FirstLine_rfe.iloc[:-1, :]
    # importance_rfe = FirstLine_rfe_Del_Target.sort_values(by="Potential (v)", ascending=False)
    importance_rfe = FirstLine_rfe_Del_Target.sort_values(by=FirstLine_rfe_Del_Target.columns.tolist()[-1],ascending=False)
    print(importance_rfe)

    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    importance_rfe.plot(kind='bar', figsize=(figure_size_xaxis, figure_size_yaxis), rot=rotationDeg,
                        fontsize=8)  # colormap='rainbow'
    plt.savefig(path+'/FeatureImportance_after.png', dpi=300, bbox_inches='tight')
    plt.close()
    return importance_rfe



#6 机器学习建模
# 6.1.1 xgboost默认超参数建模画图
# (n_estimators=2000, max_depth=100, eta=0.1, gamma=0,
# subsample=0.9, colsample_bytree=0.9, learning_rate=0.2)
def xgboost_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split


    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])




    #xgboost建模
    from xgboost import XGBRegressor
    global clf_xgboost_default
    clf_xgboost_default = XGBRegressor(n_estimators=2000, max_depth=100, eta=0.1, gamma=0,
                       subsample=0.9, colsample_bytree=0.9, learning_rate=0.2)
    clf_xgboost_default.fit(X_train, y_train)

    y_prediction=clf_xgboost_default.predict(X_test)


    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)


    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse)+'\n'+"MAE:"+str(MAE)+'\n'+"R2:"+str(R2)+'\n'+"MSE:"+str(MSE)+'\n'


    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)

    plt.tick_params(width=2)                                                  # 刻度线宽度
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)                                   # 控制主次刻度线的长度，宽度
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(10000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.axis('tight')
    plt.minorticks_on()

    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)

    #plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center',transform=ax.transAxes)
    plt.savefig(path+'/xgboost-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()




    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf_xgboost_default, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    print(type(scores))

    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)

    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    #     ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    #     xminorLocator   = MultipleLocator(1000)
    #     yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/xgboost-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()





    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf_xgboost_default.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    #plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center',transform=ax.transAxes)
    plt.savefig(path+'/xgboost-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2


# 6.1.2 xgboost自己修改超参数, 建模
# 画图得到拟合图以及交叉验证图
# (n_estimators=2000xxx, max_depth=100xxx, eta=0.1xxx, gamma=0xxx,
# subsample=0.9xxx, colsample_bytree=0.9xxx, learning_rate=0.2xxx)

def xgboost_modify(a, b, c, d, e, f, g,path,csvName):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt  # 计算准确率xgboost
    from sklearn.model_selection import train_test_split
    import pandas as pd

    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""
    data = pd.DataFrame(pd.read_csv(csvName))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    # xgboost建模
    from xgboost import XGBRegressor
    global clf_xgboost_modify
    clf_xgboost_modify = XGBRegressor(n_estimators=a, max_depth=b, eta=c, gamma=d,
                       subsample=e, colsample_bytree=f, learning_rate=g)
    clf_xgboost_modify.fit(X_train, y_train)
    Continuous_Xgboost=clf_xgboost_modify.fit(X_train, y_train)
    y_prediction = clf_xgboost_modify.predict(X_test)

    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)

    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path+'/xgboost-modify.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf_xgboost_modify, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)

    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    #     ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    #     xminorLocator   = MultipleLocator(1000)
    #     yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/xgboost_modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf_xgboost_modify.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/xgboost-modify-train-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_Xgboost, open(path+"/Continuous_Xgboost.dat", "wb"))
    return str1, scores, str2



# 6.1.3 xgboost randomSearchCV, 包含了交叉验证
def xgboost_RandomSearchCV(path):
    # 数据切分
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt  # 计算准确率xgboost
    from sklearn.model_selection import train_test_split

    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    # 尝试random search
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBRegressor

    param_distribs = {
        'n_estimators': range(80, 200, 40),
        'max_depth': range(2, 15, 4),
        'learning_rate': np.linspace(0.01, 2, 4),
        'subsample': np.linspace(0.7, 0.9, 4),
        'colsample_bytree': np.linspace(0.5, 0.98, 4),
        'min_child_weight': range(1, 9, 3)
    }

    clf = XGBRegressor()
    global rnd_search_cv_xgboost
    rnd_search_cv_xgboost = RandomizedSearchCV(clf, param_distribs, n_iter=300, cv=10, scoring='neg_mean_squared_error')
    rnd_search_cv_xgboost.fit(X_train, y_train)
    y_prediction = rnd_search_cv_xgboost.predict(X_test)

    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)

    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)

    print("RMSE:", rmse)
    print("MAE:", MAE)

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)

    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()

    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)

    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path+'/xgboost-RandomizedSearchCV.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(rnd_search_cv_xgboost, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)

    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    #     ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    #     xminorLocator   = MultipleLocator(1000)
    #     yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/Xgboost_rnd_search_cv-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()

   # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/xgboost-train-randomSearch.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2



# 6.1.4 xgboost GridSearchCV网格搜索（不随机）, 包含了交叉验证
def xgboost_GridSearchCV(path):
    # 数据切分
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt  # 计算准确率xgboost
    from sklearn.model_selection import train_test_split

    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    # 尝试random search
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor

    param_distribs = {
        'n_estimators': range(80, 200, 30),
        'max_depth': range(2, 15, 3),
        'learning_rate': np.linspace(0.01, 2, 4),
        'subsample': np.linspace(0.7, 0.9, 4),
        'colsample_bytree': np.linspace(0.5, 0.98, 4),
        'min_child_weight': range(1, 9, 3)
    }

    clf = XGBRegressor()
    grid_search_cv = GridSearchCV(clf, param_distribs, n_iter=300, cv=10, scoring='neg_mean_squared_error')
    grid_search_cv.fit(X_train, y_train)
    y_prediction = grid_search_cv.predict(X_test)

    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)

    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)

    print("RMSE:", rmse)
    print("MAE:", MAE)

    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)

    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()

    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)

    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path+'/xgboost-GridSearchCV.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(grid_search_cv, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)

    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    #     ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    #     xminorLocator   = MultipleLocator(1000)
    #     yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/grid_search_cv-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()

   # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Xgboost-grid_search_train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2




#6.2 随机森林机器学习建模
# 6.2.1 随机森林默认超参数建模画图
def RandomForest_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    #Random forest建模
    from sklearn import ensemble
    global clf_rf_default
    clf_rf_default = ensemble.RandomForestRegressor()
    clf_rf_default.fit(X_train, y_train)

    RF=clf_rf_default.fit(X_train, y_train)

    y_prediction=clf_rf_default.predict(X_test)

    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)

    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()

    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)

    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/randomForest-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()

    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf_rf_default, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)

    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    #     ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    #     xminorLocator   = MultipleLocator(1000)
    #     yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/randomForest-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()


    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf_rf_default.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/randomForest-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2



# 6.2.2 Random forest modify 自己修改超参数, 建模
def RandomForest_modify(a, b, c, d, e,path,csvname):
# max_depth, max_features, min_samples_split, n_estimators, random_state
# 20, 0.3, 2, 10, 10
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # RandomForest建模
    from sklearn import ensemble
    clf = ensemble.RandomForestRegressor(max_depth=a,max_features=b, min_samples_split=c,n_estimators=d,random_state=e)
    clf.fit(X_train, y_train)
    Continuous_RF=clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    #看是否有预测集
    # if xxx:
    #     pass
    # else:
    #     print(clf.predict(input))
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse)+'\n'+"MAE:"+str(MAE)+'\n'+"R2:"+str(R2)+'\n'+"MSE:"+str(MSE)+'\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path+'/RandomForest-modify.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:",scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i+1,"scores_mean:",scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/RandomForest_modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
            + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/RandomForest-modify-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_RF, open(path+"/Continuous_RF.dat", "wb"))
    return str1,scores,str2

# 6.2.3 RandomForest randomSearchCV, 包含了交叉验证
def RandomForest_RandomSearchCV(path):
    # 数据切分
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 尝试random search
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn import ensemble
    param_distribs = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 200, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [130, 180, 230]}
    clf = ensemble.RandomForestRegressor()
    global rnd_search_cv_Random_forest
    rnd_search_cv_Random_forest = RandomizedSearchCV(clf, param_distribs, n_iter=300, cv=10, scoring='neg_mean_squared_error')
    rnd_search_cv_Random_forest.fit(X_train, y_train)
    y_prediction = rnd_search_cv_Random_forest.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path+'/RandomForest-RandomizedSearchCV.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(rnd_search_cv_Random_forest, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/RandomForest_rnd_search_cv-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
   # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/RandomForest-train-randomSearchCV.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2



#6.3 bagging机器学习建模
# 6.3.1 bagging默认超参数建模画图

from sklearn import ensemble
def Bagging_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    clf = ensemble.BaggingRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Bagging-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/Bagging-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Bagging-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.3.2 bagging自定义超参数建模画图
def Bagging_modify(a, b, c,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    # 数据切分
    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    clf = ensemble.BaggingRegressor(n_estimators=a,max_samples=b, max_features=c)
    clf.fit(X_train, y_train)
    Continuous_Bagging=clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path + '/Bagging-modify.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path + '/Bagging-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:", R2_train)
    print("MSE:", MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train),
             fontproperties='Times New Roman', size=20, horizontalalignment='center')
    plt.savefig(path + '/Bagging-modify-train.png', dpi=300, bbox_inches='tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_Bagging, open(path + "/Continuous_Bagging.dat", "wb"))
    return str1, scores, str2


#6.4 AdaBoost机器学习建模
# 6.4.1 AdaBoost默认超参数建模画图

from sklearn import ensemble
def AdaBoost_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    clf = ensemble.AdaBoostRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/AdaBoost-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/AdaBoost-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/AdaBoost-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.4.2 AdaBoost自定义超参数建模画图
def AdaBoost_modify(a, b, c,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    # 数据切分
    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble


    if c==0.3:
        loss1='exponential'
    elif c==0.2:
        loss1 = 'square'
    else:
        loss1 = 'linear'



    clf = ensemble.AdaBoostRegressor(n_estimators=a,learning_rate=b,loss=loss1)
    clf.fit(X_train, y_train)
    Continuous_AdaBoost=clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path + '/AdaBoost-modify.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path + '/AdaBoost-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:", R2_train)
    print("MSE:", MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train),
             fontproperties='Times New Roman', size=20, horizontalalignment='center')
    plt.savefig(path + '/AdaBoost-modify-train.png', dpi=300, bbox_inches='tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_AdaBoost, open(path + "/Continuous_AdaBoost.dat", "wb"))
    return str1, scores, str2


#6.5 GradientBoosting机器学习建模
# 6.5.1 GradientBoosting默认超参数建模画图

from sklearn import ensemble
def GradientBoosting_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    clf = ensemble.GradientBoostingRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/GradientBoosting-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/GradientBoosting-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/GradientBoosting-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.5.2 GradientBoosting自定义超参数建模画图
def GradientBoosting_modify(a, b, c,d,e,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    # (n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2,'min_samples_leaf': 1,'learning_rate': 0.1)
    clf = ensemble.GradientBoostingRegressor(n_estimators=a, max_depth= b,min_samples_split=c,min_samples_leaf=int(d),learning_rate= e)
    clf.fit(X_train, y_train)
    Continuous_GradientBoosting = clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path + '/GradientBoosting-modify.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path + '/GradientBoosting-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:", R2_train)
    print("MSE:", MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train),
             fontproperties='Times New Roman', size=20, horizontalalignment='center')
    plt.savefig(path + '/GradientBoosting-modify-train.png', dpi=300, bbox_inches='tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_GradientBoosting, open(path + "/Continuous_GradientBoosting.dat", "wb"))
    return str1, scores, str2

#6.6 ExtraTree机器学习建模
# 6.6.1 ExtraTree默认超参数建模画图

def ExtraTree_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.tree import ExtraTreeRegressor
    clf = ExtraTreeRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/ExtraTree-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/ExtraTree-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/ExtraTree-modify-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.6.2 ExtraTree自定义超参数建模画图
def ExtraTree_modify(a, b, c,e,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn.tree import ExtraTreeRegressor

    if a==0:
        max_depth1=None
    else:
        max_depth1=a
    if b==0.1:
        max_features1='sqrt'
    elif b==0.2:
        max_features1 = 'log2'
    elif b == 0:
        max_features1 = None
    elif b == 0.3:
        max_features1 = 'auto'
    else:
        max_features1 = b
    if e==0:
        random_state1=None
    else:
        random_state1=e
    # max_depth=None,max_features['sqrt', 'log2', None,'auto']='auto',min_samples_split=2,random_state=None
    clf = ExtraTreeRegressor(max_depth=max_depth1,max_features=max_features1,min_samples_split=c,
                             random_state=random_state1)
    clf.fit(X_train, y_train)
    Continuous_ExtraTree = clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig(path + '/ExtraTree-modify.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path + '/ExtraTree-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:", R2_train)
    print("MSE:", MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train),
             fontproperties='Times New Roman', size=20, horizontalalignment='center')
    plt.savefig(path + '/ExtraTree-modify-train.png', dpi=300, bbox_inches='tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_ExtraTree, open(path+"/Continuous_ExtraTree.dat", "wb"))
    return str1, scores, str2


# 6.7 svm机器学习建模

# 6.7.1 svm默认超参数建模画图
def svm_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn import svm
    clf = svm.SVR()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/svm-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/svm-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/svm-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.7.2 Svm自定义超参数建模画图
def Svm_modify(a, b,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn import svm
    #C=1.0, epsilon=0.1
    clf = svm.SVR(C=a, epsilon=b)
    clf.fit(X_train, y_train)
    Continuous_Svm = clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Svm-modify.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/Svm-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Svm-modify-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_Svm, open(path + "/Continuous_Svm.dat", "wb"))
    return str1, scores, str2



# 6.8 DecisionTree机器学习建模

# 6.8.1 DecisionTree默认超参数建模画图
def DecisionTree_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/DecisionTree-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/DecisionTree-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/DecisionTree-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.8.2 DecisionTree自定义超参数建模画图
def DecisionTree_modify(a, b,c,d,e,f,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn import tree
    # max_depth=None,max_features=None,min_samples_split=2,min_samples_leaf=1,random_state=None,max_leaf_nodes=None
    if a==0:
        max_depth1=None
    else:
        max_depth1=a
    if b==0.1:
        max_features1='sqrt'
    elif b==0.2:
        max_features1 = 'log2'
    elif b == 0:
        max_features1 = None
    elif b == 0.3:
        max_features1 = 'auto'
    else:
        max_features1 = b
    if e==0:
        random_state1=None
    else:
        random_state1=e
    if f==0:
        max_leaf_nodes1=None
    else:
        max_leaf_nodes1=f
    clf = tree.DecisionTreeRegressor(max_depth=max_depth1,max_features=max_features1,min_samples_split=c,
                                     min_samples_leaf=d,random_state=random_state1,max_leaf_nodes=max_leaf_nodes1)
    clf.fit(X_train, y_train)
    Continuous_DecisionTree = clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/DecisionTree-modify.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/DecisionTree-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/DecisionTree-modify-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_DecisionTree, open(path + "/Continuous_DecisionTree.dat", "wb"))
    return str1, scores, str2



# 6.9 LinearRegression机器学习建模

# 6.9.1 LinearRegression默认超参数建模画图
def LinearRegression_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # datasave = pd.DataFrame([[y_test], [y_prediction]])
    # datasave = pd.DataFrame({'y_test': y_test, 'y_prediction': y_prediction})
    # datasave.to_csv("LR-test.csv")
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/LinearRegression-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/LinearRegression-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/LinearRegression-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.9.2 LinearRegression自定义超参数建模画图
def LinearRegression_modify(a, b,c,d,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.linear_model import LinearRegression
    # fit_intercept=True, normalize=False, copy_X=True, n_jobs=None
    if a==0:
        fit_intercept1=False
    else:
        fit_intercept1 = True
    if b==0:
        normalize1=False
    else:
        normalize1 = True
    if c==0:
        copy_X1=False
    else:
        copy_X1 = True
    if d==0:
        n_jobs1=None
    else:
        n_jobs1 = d
    clf = LinearRegression(fit_intercept=fit_intercept1, normalize=normalize1, copy_X=copy_X1, n_jobs=n_jobs1)
    clf.fit(X_train, y_train)

    Continuous_LinearRegression = clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # datasave = pd.DataFrame([[y_test], [y_prediction]])
    # datasave = pd.DataFrame({'y_test': y_test, 'y_prediction': y_prediction})
    # datasave.to_csv("LR-test.csv")
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/LinearRegression-modify.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/LinearRegression-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/LinearRegression-modify-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_LinearRegression, open(path + "/Continuous_LinearRegression.dat", "wb"))
    return str1, scores, str2


# 6.10 Ridge机器学习建模

# 6.10.1 Ridge默认超参数建模画图
def Ridge_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.linear_model import Ridge
    clf = Ridge()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Ridge-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/Ridge-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Ridge-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2

# 6.10.2 Ridge自定义超参数建模画图
def Ridge_modify(a, b,c,d,e,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.linear_model import Ridge
    # alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, random_state=None
    if b==0:
        fit_intercept1=False
    else:
        fit_intercept1 = True
    if c==0:
        normalize1=False
    else:
        normalize1 = True
    if d==0:
        copy_X1=False
    else:
        copy_X1 = True
    if e==0:
        random_state1=None
    else:
        random_state1=e
    clf = Ridge(alpha=a, fit_intercept=fit_intercept1, normalize=normalize1, copy_X=copy_X1, random_state=random_state1)
    clf.fit(X_train, y_train)
    Continuous_Ridge = clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Ridge-modify.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/Ridge-modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/Ridge-modify-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_Ridge, open(path + "/Continuous_Ridge.dat", "wb"))
    return str1, scores, str2


# 6.11 MLP机器学习建模

# 6.11.1 MLP默认超参数建模画图
def MLP_default(path):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/MLP-default.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/MLP-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/MLP-default-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    return str1, scores, str2


# 6.11.2 MLP_modify手动修改超参数建模画图
def MLP_modify(l,a,m,ha,hb,path,csvname):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    """X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]"""

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.neural_network import MLPRegressor
    # 0.01,0.0001,200000,200,200
    clf = MLPRegressor(solver='lbfgs', activation='relu', learning_rate_init=l, alpha=a, max_iter=m,
                 hidden_layer_sizes=(ha, hb))

    Continuous_MLP = clf.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    str1 = "RMSE:" + str(rmse) + '\n' + "MAE:" + str(MAE) + '\n' + "R2:" + str(R2) + '\n' + "MSE:" + str(MSE) + '\n'

    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/MLP_modify.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig(path+'/MLP_modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.close()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    str2 = "RMSE:" + str(rmse_train) + '\n' + "MAE:" + str(MAE_train) + '\n' + "R2:" + str(R2_train) + '\n' \
           + "MSE:" + str(MSE_train) + '\n'

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig(path+'/MLP_modify-train.png', dpi=300, bbox_inches = 'tight')
    plt.close()
    import pickle
    pickle.dump(Continuous_MLP, open(path + "/Continuous_MLP.dat", "wb"))
    return str1, scores, str2




# 7 基于给定模型预测
def model_modify_predict(csvName,path,model_path):
    """import pandas as pd
    Predict_features = pd.DataFrame(pd.read_csv(csvName))
    featureData1 = Predict_features.values[:, :]
    # StandardScaler.fit(featureData1)
    # featureData2 = StandardScaler.transform(featureData1)
    # print(featureData2)
    predict = clf_xgboost_modify.predict(featureData1)
    predict_Ef = pd.DataFrame(predict)
    predict_Ef.to_csv(path + "/Predict_xgboost_dataset_modify.csv")"""

    import pickle
    import os
    import pandas as pd
    import numpy as np
    import csv
    from sklearn import preprocessing
    loaded_model = pickle.load(open(model_path, "rb"))
    data = pd.DataFrame(pd.read_csv(csvName))
    X = pd.DataFrame(pd.read_csv(csvName)).values[:, :]

    # 特征缩放，映射到0和1之间的范围
    for i in range(X.shape[1]):
        X[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X[:, [i]])
    target = loaded_model.predict(X)
    print(loaded_model.predict(X))
    file_name = "prediction_output("+os.path.basename(model_path)[:-4]+").csv"

    tg = pd.DataFrame(target,columns=["Output"])

    prediction = pd.concat([data, tg], axis=1)
    prediction.to_csv(path +"/"+ file_name)
    return path +"/"+ file_name





# 7.1.1 预测集基于xgboost_default()
# 画图得到拟合图以及交叉验证图
# (n_estimators=2000xxx, max_depth=100xxx, eta=0.1xxx, gamma=0xxx,
# subsample=0.9xxx, colsample_bytree=0.9xxx, learning_rate=0.2xxx)
"""def xgboost_default_predict(csvName,path):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # xgboost建模
    from xgboost import XGBRegressor
    clf = XGBRegressor()
    clf.fit(X_train, y_train)
    # 需要准备新的待预测的特征集x_New.csv(不含目标列), 导入 x_New的列数为之前设置的rfe剩余特征个数
    import pandas as pd
    x_New = pd.read_csv(csvName)
    print("new features dataset: ", x_New)
    # xgboost_modify新的预测
    y_New_prediction = clf.predict(x_New)
    y_New_prediction = pd.DataFrame(y_New_prediction)
    y_New_prediction.columns = ['Output']
    print("new output: ", y_New_prediction)
    NewData = pd.concat([x_New, y_New_prediction], axis=1)
    print("New total Data: ", NewData)
    NewData.to_csv(path+"/New_prediction_total_xgboost_default.csv")
    return x_New,y_New_prediction, NewData"""

def xgboost_default_predict(csvName,path):
    import pandas as pd
    Predict_features = pd.DataFrame(pd.read_csv(csvName))
    featureData1 = Predict_features.values[:, :]
    # StandardScaler.fit(featureData1)
    # featureData2 = StandardScaler.transform(featureData1)
    # print(featureData2)
    predict = clf_xgboost_default.predict(featureData1)
    predict_Ef = pd.DataFrame(predict)
    predict_Ef.to_csv(path + "/Predict_xgboost_dataset.csv")


# 7.1.2 预测集基于xgboost_modify
# 画图得到拟合图以及交叉验证图
# (n_estimators=2000xxx, max_depth=100xxx, eta=0.1xxx, gamma=0xxx,
# subsample=0.9xxx, colsample_bytree=0.9xxx, learning_rate=0.2xxx)
"""def xgboost_modify_predict(a, b, c, d, e, f, g, csvName,path):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    try:
        X = s_rfe
        y = target
        X = X.values[:, :]
        y = y.values[:, :]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        # 数据归一化
        for i in range(X_train.shape[1]):
            X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
        for i in range(X_test.shape[1]):
            X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
        # xgboost建模
        from xgboost import XGBRegressor
        clf = XGBRegressor(n_estimators=a, max_depth=b, eta=c, gamma=d,
                           subsample=e, colsample_bytree=f, learning_rate=g)
        clf.fit(X_train, y_train)
        # 需要准备新的待预测的特征集x_New.csv(不含目标列), 导入 x_New的列数为之前设置的rfe剩余特征个数
        import pandas as pd
        x_New = pd.read_csv(csvName)
        print("new features dataset: ", x_New)
        # xgboost_modify新的预测
        y_New_prediction = clf.predict(x_New)
        y_New_prediction = pd.DataFrame(y_New_prediction)
        y_New_prediction.columns = ['Output']
        print("new output: ", y_New_prediction)
        NewData = pd.concat([x_New, y_New_prediction], axis=1)
        print("New total Data: ", NewData)
        NewData.to_csv(path+"/New_prediction_total_xgboost_modify.csv")
        return x_New,y_New_prediction, NewData

    except Exception as e:
        print(e)"""
def xgboost_modify_predict(csvName,path):
    """import pandas as pd
    Predict_features = pd.DataFrame(pd.read_csv(csvName))
    featureData1 = Predict_features.values[:, :]
    # StandardScaler.fit(featureData1)
    # featureData2 = StandardScaler.transform(featureData1)
    # print(featureData2)
    predict = clf_xgboost_modify.predict(featureData1)
    predict_Ef = pd.DataFrame(predict)
    predict_Ef.to_csv(path + "/Predict_xgboost_dataset_modify.csv")"""

    import pickle
    import pandas as pd
    import numpy as np
    import csv
    from sklearn import preprocessing
    loaded_model = pickle.load(open("Continuous_Xgboost.dat", "rb"))
    data = pd.DataFrame(pd.read_csv(csvName))
    X = data.values[:, :]
    for i in range(X.shape[1]):
        X[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X[:, [i]])
    target = loaded_model.predict(X)
    print(loaded_model.predict(X))


# 7.1.3 预测集基于rnd_search_cv_xgboost
# 画图得到拟合图以及交叉验证图
def rnd_search_cv_xgboost_predict(csvName,path):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 需要准备新的待预测的特征集x_New.csv(不含目标列), 导入 x_New的列数为之前设置的rfe剩余特征个数
    import pandas as pd
    x_New = pd.read_csv(csvName)
    print("new features dataset: ", x_New)
    # xgboost_modify新的预测
    y_New_prediction = rnd_search_cv_xgboost.predict(x_New)
    y_New_prediction = pd.DataFrame(y_New_prediction)
    y_New_prediction.columns = ['Output']
    print("new output: ", y_New_prediction)
    NewData = pd.concat([x_New, y_New_prediction], axis=1)
    print("New total Data: ", NewData)
    NewData.to_csv(path+"/New_prediction_total_rnd_search_cv_xgboost.csv")

    return x_New,y_New_prediction, NewData

# 7.2.1 预测集基于random_forest_default
def randomforest_default_predict(csvName,path):
    import pandas as pd
    Predict_features = pd.DataFrame(pd.read_csv(csvName))
    featureData1 = Predict_features.values[:, :]
    # StandardScaler.fit(featureData1)
    # featureData2 = StandardScaler.transform(featureData1)
    # print(featureData2)
    predict = clf_rf_default.predict(featureData1)
    predict_Ef = pd.DataFrame(predict)
    predict_Ef.to_csv(path + "/Predict_rf_dataset.csv")

# 8 描述符导入
# 8.1 有机分子描述符导入（NJmatML提供了pydel描述符和rdkit描述符）
# 8.1.1 pydel描述符
# 8.1.1.1 导入有机分子smiles码的csv文件


def smiles_csv_pydel(name2):
    import pandas as pd
    global data2
    data2 = pd.read_csv(name2)
    print(data2.iloc[:,0])
    return data2

# 8.1.1.2 pydel描述符生成
def pydel_featurizer(path):
    from padelpy import from_smiles
    import pandas as pd
    data2a = data2.iloc[:,0].map(lambda x : from_smiles(x).values())
    data2a = pd.DataFrame(data2a)
    data2b = data2a.iloc[:,0].apply(pd.Series)
    #写入列名
    data2c = data2.iloc[:,0].map(lambda x : from_smiles(x).keys())
    col2c = data2c.iloc[0]
    data2b.columns = col2c
    print(data2b)
    # 特征存入pydel_featurizer.csv
    data2b.to_csv(path+"/pydel_featurizer_output.csv")
    return data2b

# !pip install padelpy
# from padelpy import from_smiles
# import pandas as pd
# # calculate molecular descriptors for propane
# CCC_descriptors = from_smiles('CCC')
# print(CCC_descriptors)
# print(CCC_descriptors['nAcid'])
# print(CCC_descriptors['ALogP'])
# print(CCC_descriptors['ALogp2'])


# 8.1.2 rdkit描述符
# 8.1.2.1 导入有机分子smiles码的csv文件
def smiles_csv_rdkit(name3):
    import pandas as pd
    global data3
    data3 = pd.read_csv(name3)
    print(data3.iloc[:,0])
    return data3



# 8.1.2.2 rdkit描述符生成
def rdkit_featurizer(path):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import rdDepictor
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    # choose 200 molecular descriptors
    chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
    # create molecular descriptor calculator
    mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)
    data4 = data3.iloc[:,0].map(lambda x : mol_descriptor_calculator.CalcDescriptors(Chem.MolFromSmiles(x)))
    data4 = pd.DataFrame(data4)
    data5 = pd.DataFrame()
    # split to 200 columns
    for i in range(0, 200):
        data5 = pd.concat([data5, data4.applymap(lambda x: x[i])], axis=1)
    data5.columns = chosen_descriptors
    print(data5)
    # 特征存入rdkit_featurizer.csv
    data5.to_csv(path+"/rdkit_featurizer_output.csv")
    return data5

# 8.1.2.3 从smiles码画分子
def drawMolecule(smiles):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import rdDepictor
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    m = Chem.MolFromSmiles(smiles)
    print(m)

# 8.1.2.3 从smiles码画分子
# drawMolecule('Cc1ccccc1') #括号里（SMILES码两边）请加引号


# 8.2 无机材料描述符 (NJmatML参考Matminer使用类独热编码方式特征化无机化学式)
# 8.2.1 导入含有无机材料化学式的csv
def inorganic_csv(name4):
    import pandas as pd
    global data4
    data4 = pd.read_csv(name4)
    print(data4)
    return data4

# 8.2.1 用于magpie,导入含有无机材料化学式的csv
def inorganic_magpie_csv(name20):
    import pandas as pd
    global data20
    data20 = pd.read_csv(name20)
    print(data20)
    return data20

# 8.2.2 matminer无机材料（类独热编码）描述符生成，102维
# 例如(Fe2AgCu2)O3, Fe2O3, Cs3PbI3, MoS2, CuInGaSe, Si, TiO2等
def inorganic_featurizer(path):
    import pandas as pd
    from matminer.featurizers.composition.element import ElementFraction
    from pymatgen.core import Composition
    ef = ElementFraction()
    list4 = list(map(lambda x: Composition(x), data4.iloc[:,0]))
    data7 = pd.DataFrame()
    for i in range(0, len(data4.index)):
        data7 = pd.concat([data7, pd.DataFrame(ef.featurize(list4[i])).T])
    data8 = data7.reset_index()
    data8 = data8.iloc[:, 1:]
    print(data8)
    element_fraction_labels = ef.feature_labels()
    print(element_fraction_labels)
    # 特征存入pydel_featurizer.csv
    data8.to_csv(path+"/inorganic_featurizer_output.csv",index=None)
    return data8,element_fraction_labels

# 8.2.3 magpie（matminer)无机材料描述符生成
def inorganic_magpie_featurizer(path):
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import ElementProperty
    import pandas as pd

    str_to_comp = StrToComposition(target_col_id='composition')
    df_comp = str_to_comp.featurize_dataframe(data20, col_id='formula')   #此处规定csv中第一列列名必须是 formula  否则软件闪退！！！！！
    features = ['Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT',
                'Column', 'Row', 'CovalentRadius', 'Electronegativity',
                'NsValence', 'NpValence', 'NdValence', 'NfValence', 'NValence',
                'NsUnfilled', 'NpUnfilled', 'NdUnfilled', 'NfUnfilled', 'NUnfilled',
                'GSvolume_pa', 'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']

    stats = ['mean', 'minimum', 'maximum', 'range', 'avg_dev', 'mode']

    featurizer = ElementProperty(data_source='magpie',
                                 features=features,
                                 stats=stats)
    df_features = featurizer.featurize_dataframe(df_comp, col_id='composition')
    df_features = df_features.iloc[:, 2:-1]
    print(df_features)
    df_features.to_csv(path+"/1_magpie.csv",index=None)
    return df_features


# 9 遗传算法设计新特征
## 9.1 普通默认运算符
def gp_default(r_thresh):  ## 输入参数为皮尔森阈值 ：例如输入0.6后，大于0.6的才显示
    import numpy as np
    from sklearn import preprocessing
    from gplearn import genetic
    X = data.values[:,:-1]
    y = data.values[:,-1]
    for i in range(X.shape[1]):
        X[:,[i]] = preprocessing.MinMaxScaler().fit_transform(X[:,[i]])
    est_gp = genetic.SymbolicTransformer(population_size=1000,
                               generations=91, stopping_criteria=0.01,
                               p_crossover=0.8, p_subtree_mutation=0.05,
                               p_hoist_mutation=0.05, p_point_mutation=0.05,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=None,n_components=100)
    V=est_gp.fit(X, y)
    px=V.transform(X)
    str1=""
    for i in range(0,50):
        pear=np.corrcoef(px[:,i], y)
        pea=pear[0,1]
        if pea>r_thresh:
            print(pea,i,V[i])
            str1=str1+str(pea)+"  "+str(i)+"  "+str(V[i])+"\n"
    print('\n***************************')
    for i in range(len(data.columns.values.tolist())):
        print(i, data.columns.values.tolist()[i])
    return str1,data

## 9.2 更多运算符
def gp_tan(r_thresh):
    import numpy as np
    from sklearn import preprocessing
    from gplearn import genetic
    X = data.values[:, :-1]
    y = data.values[:, -1]
    for i in range(X.shape[1]):
        X[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X[:, [i]])
    function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg','inv','sin','cos','tan', 'max', 'min']
    est_gp = genetic.SymbolicTransformer(population_size=1000,
                               generations=91, stopping_criteria=0.01,
                               p_crossover=0.8, p_subtree_mutation=0.05,
                               p_hoist_mutation=0.05, p_point_mutation=0.05,
                               max_samples=0.9, verbose=1,function_set=function_set,
                               parsimony_coefficient=0.01, random_state=None,n_components=100)
    V=est_gp.fit(X, y)
    px=V.transform(X)
    str1 = ""
    for i in range(0,50):
        pear=np.corrcoef(px[:,i], y)
        pea=pear[0,1]
        if pea>r_thresh:
            print(pea,i,V[i])
            str1 = str1 + str(pea) + "  " + str(i) + "  " + str(V[i]) + "\n"
    print('\n***************************')
    for i in range(len(data.columns.values.tolist())):
        print(i, data.columns.values.tolist()[i])
    return str1, data

## 9.3 tSR默认形式为(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
def tSR_default(r_thresh,path):
    import numpy as np
    X = data_rfe.values[:, :-1]
    y = data_rfe.values[:, -1]
    for i in range(0,(data_rfe.shape[1]-1)):
         for j in range(0,(data_rfe.shape[1]-1)):
              for k in range(0,(data_rfe.shape[1]-1)):
                    for n in range(0,(data_rfe.shape[1]-1)):
                         px=(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
                         per=np.corrcoef(px, y)
                         if per[0,1]>r_thresh or per[0,1]< -1 * r_thresh:
                              print(per[0,1])
                              print(i,j,k,n)
                              print(data_rfe.columns.values.tolist()[i],data_rfe.columns.values.tolist()[j],data_rfe.columns.values.tolist()[k],data_rfe.columns.values.tolist()[n])
                              print('(',data_rfe.columns.values.tolist()[i],'-',data_rfe.columns.values.tolist()[j],')','*','(',data_rfe.columns.values.tolist()[k],'-',data_rfe.columns.values.tolist()[n],')')
                              print('**********************************************')
                              with open(path+"/data.txt", "a+") as f:
                                  f.write(str(per[0,1])+"\n")
                                  f.write(str(i)+" "+str(j)+" "+str(k)+" "+str(n)+"\n")
                                  f.write(str(data_rfe.columns.values.tolist()[i])+" "
                                          +str(data_rfe.columns.values.tolist()[j])+" "
                                          +str(data_rfe.columns.values.tolist()[k])+" "
                                          +str(data_rfe.columns.values.tolist()[n])+"\n")
                                  f.write("( "+str(data_rfe.columns.values.tolist()[i]) + " - "
                                          + str(data_rfe.columns.values.tolist()[j]) + " ) * ("
                                          + str(data_rfe.columns.values.tolist()[k]) + " - "
                                          + str(data_rfe.columns.values.tolist()[n]) + " )\n")
                                  f.write('**********************************************\n')

# ## 9.4 tSR更多运算符，默认形式为(X[:,i]-X[:,j])*(X[:,k]-X[:,n]) 可能删去，没有用处
# def tSR_tan(r_thresh):
#     import numpy as np
#     X = data.values[:, :-1]
#     y = data.values[:, -1]
#     for i in range(0,(data.shape[1]-1)):
#      for j in range(0,(data.shape[1]-1)):
#       for k in range(0,(data.shape[1]-1)):
#         for n in range(0,(data.shape[1]-1)):
#          px=(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
#          per=np.corrcoef(px, y)
#          if per[0,1]>r_thresh or per[0,1] < -1 * r_thresh:
#           print(per[0,1])
#           print(i,j,k,n)
#           print(data.columns.values.tolist()[i],data.columns.values.tolist()[j],data.columns.values.tolist()[k],data.columns.values.tolist()[n])
#           print('(',data.columns.values.tolist()[i],'-',data.columns.values.tolist()[j],')','*','(',data.columns.values.tolist()[k],'-',data.columns.values.tolist()[n],')')
#           print('**********************************************')

#未完待续（其他机器学习算法，网格搜索，预测集建立，描述符填充等等）


# 10.1
def randomforest_Classifier(a, b, c, d, e, f,path,csvName):
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
    import pickle

    data = pd.DataFrame(pd.read_csv(csvName))


    X = data.values[:, 1:-1]
    y = data.values[:, -1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    param_grid = {
        'n_estimators': [50, 80, 100, 120],
        'max_depth': [6, 7],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1, 2],
        'random_state': [0, 1]
    }

    # Create a RandomForestClassifier object
    rfc = RandomForestClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    str1 = f"Best parameters: {grid_search.best_params_}" + "\n" + f"Best score: {grid_search.best_score_}"


    clf = RandomForestClassifier(max_depth=a, random_state=b, min_samples_leaf=c, max_features=d, min_samples_split=e,
                                 n_estimators=f)
    clf.fit(X, y)
    Classified_two_RF = clf.fit(X, y)

    # 画出ROC曲线 RandomForest test
    y_score = clf.fit(X, y).predict_proba(X_test)
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
    #plt.show()
    plt.savefig(path + '/RandomForest_test_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 RandomForest test
    clf.fit(X, y)
    prey = clf.predict(X_test)
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
    #plt.show()
    plt.savefig(path + '/RandomForest_test_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str2 = "fpr:"+str(fpr) + '\n' + "tpr:"+str(tpr)+"\n"+"true:"+str(true)

    # 画出ROC曲线 RandomForest train的AUC
    y_score = clf.fit(X, y).predict_proba(X_train)
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
    #plt.show()
    plt.savefig(path + '/RandomForest_train_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 RandomForest train 混淆矩阵
    clf.fit(X, y)
    prey = clf.predict(X_train)
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
    #plt.show()
    plt.savefig(path + '/RandomForest_train_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str3 = "fpr:"+str(fpr) + '\n' + "tpr:"+str(tpr)+"\n"+"true:"+str(true)

    pickle.dump(Classified_two_RF, open(path+"/Classified_two_RF.dat", "wb"))

    return str1,str2,str3

# 10.2
def extratrees_classifier(a, b, c, d, e, f,path,csvName):
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
    import pickle

    data = pd.DataFrame(pd.read_csv(csvName))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    from sklearn.ensemble import ExtraTreesClassifier

    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score, make_scorer

    # Generate some data for classification
    X, y = X_train, y_train

    # Define the ExtraTreesClassifier
    et_clf = ExtraTreesClassifier()

    # Define the parameters to search
    params = {
        'n_estimators': [2, 50, 100, 200],
        'max_depth': [None, 5, 7, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Define the scoring metric
    scoring = make_scorer(accuracy_score)

    # Define the grid search with cross-validation
    grid_search = GridSearchCV(et_clf, params, scoring=scoring, cv=5, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X, y)

    # Print the best parameters and best score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)
    str1 = f"Best parameters: {grid_search.best_params_}" + "\n" + f"Best score: {grid_search.best_score_}"

    if b==0:
        max_depth1=None
    else:
        max_depth1=b
    if e==0.1:
        max_features1='sqrt'
    elif e==0.2:
        max_features1 = 'log2'
    elif e == 0:
        max_features1 = None
    else:
        max_features1 = e
    clf = ExtraTreesClassifier(n_estimators=a, max_depth=max_depth1, min_samples_split=c, random_state=d, max_features=max_features1,
                               min_samples_leaf=f)
    clf.fit(X, y)
    Classified_two_ExtraTrees = clf.fit(X, y)
    # 画出ROC曲线 ExtraTrees test
    y_score = clf.fit(X, y).predict_proba(X_test)
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(path + '/ExtraTrees_test_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 ExtraTrees
    clf.fit(X, y)
    prey = clf.predict(X_test)
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
    #plt.show()
    plt.savefig(path + '/ExtraTrees_test_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str2 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

    # 画出ROC曲线 ExtraTrees train
    y_score = clf.fit(X, y).predict_proba(X_train)
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(path + '/ExtraTrees_train_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 RandomForest train 混淆矩阵
    clf.fit(X, y)
    prey = clf.predict(X_train)
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
    #plt.show()
    plt.savefig(path + '/ExtraTrees_train_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str3 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

    import pickle
    pickle.dump(Classified_two_ExtraTrees, open(path + "/Classified_two_ExtraTrees.dat", "wb"))
    return str1, str2, str3

# 10.3
def GaussianProcess_classifier(a, b, c, d, e,path,csvName):
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
    import pickle

    data = pd.DataFrame(pd.read_csv(csvName))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    from sklearn.datasets import make_classification

    # Define the parameter grid to search over
    param_grid = {
        "kernel": [1.0 * RBF(length_scale=1.0)],
        "optimizer": ['fmin_l_bfgs_b'],
        "n_restarts_optimizer": [0, 1, 2],
        "max_iter_predict": [5, 10, 30, 40, 50, 100]
    }

    # Create a GaussianProcessClassifier object
    clf = GaussianProcessClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters found:", grid_search.best_params_)

    # Evaluate the best estimator on the test data
    print("Test accuracy:", grid_search.score(X_test, y_test))

    str1 = f"Best parameters: {grid_search.best_params_}" + "\n" + f"Test accuracy: {grid_search.score(X_test, y_test)}"

    if e==0:
        clf = GaussianProcessClassifier(kernel=a * RBF(length_scale=b), max_iter_predict=c, n_restarts_optimizer=d,
                                        optimizer='fmin_l_bfgs_b')
    Classified_two_GaussianProcess = clf.fit(X, y)

    # 画出ROC曲线 GaussianProcess test
    y_score = clf.fit(X, y).predict_proba(X_test)
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

    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize=20, frameon=False)
    #plt.show()
    plt.savefig(path + '/GaussianProcess_test_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 GaussianProcess test
    clf.fit(X, y)
    prey = clf.predict(X_test)
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
    #plt.show()
    plt.savefig(path + '/GaussianProcess_test_CM.png', dpi=300, bbox_inches='tight')
    print("true:", true)
    str2 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

    # 画出ROC曲线 GaussianProcess train
    y_score = clf.fit(X, y).predict_proba(X_train)
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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize=20, frameon=False)
    #plt.show()
    plt.savefig(path + '/GaussianProcess_train_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 GaussianProcess train
    clf.fit(X, y)
    prey = clf.predict(X_train)
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
    #plt.show()
    plt.savefig(path + '/GaussianProcess_train_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str3 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

    import pickle
    pickle.dump(Classified_two_GaussianProcess, open(path + "/Classified_two_GaussianProcess.dat", "wb"))
    return str1, str2, str3

# 10.4
def KNeighbors_classifier(a, b, c, d, e, f,csvName,path):
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
    import pickle

    data = pd.DataFrame(pd.read_csv(csvName))

    X = data.values[:, :-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    clf = KNeighborsClassifier(n_neighbors=8)
    clf.fit(X, y)

    # 画出ROC曲线 KNeighbors test
    y_score = clf.fit(X, y).predict_proba(X_test)
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
    plt.show()

    # 画出混淆矩阵 KNeighbors test
    clf.fit(X, y)
    prey = clf.predict(X_test)
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
    plt.show()
    print("true:", true)

    from sklearn.metrics import accuracy_score
    score = []
    for K in range(40):
        K_value = K + 1
        knn = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score.append(round(accuracy_score(y_test, y_pred) * 100, 2))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 41), score, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('The Learning curve')
    plt.xlabel('K Value')
    plt.ylabel('Score')

# 10.5
def DecisionTree_classifier(a, b, c, d, e,path,csvName):
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
    import pickle

    data = pd.DataFrame(pd.read_csv(csvName))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Define the parameter grid to search over
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    # Create a DecisionTreeClassifier object
    dtc = DecisionTreeClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(dtc, param_grid=param_grid, cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters found:", grid_search.best_params_)

    # Evaluate the best estimator on the test data
    print("Test accuracy:", grid_search.score(X_test, y_test))

    str1 = f"Best parameters: {grid_search.best_params_}" + "\n" + f"Test accuracy: {grid_search.score(X_test, y_test)}"

    if a==0.1:
        criterion='gini'
    else:
        criterion = 'entropy'

    if b==0:
        max_depth1=None
    else:
        max_depth1 = b
    if c==0.1:
        max_features_1='auto'
    elif c==0.2:
        max_features_1 = 'sqrt'
    elif c==0.3:
        max_features_1 = 'log2'
    else:
        max_features_1 =None

    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth1, max_features=max_features_1, min_samples_leaf=d,
                                 min_samples_split=e)
    clf.fit(X, y)
    Classified_two_DecisionTree = clf.fit(X, y)

    # 画出ROC曲线 DecisionTree test
    y_score = clf.fit(X, y).predict_proba(X_test)
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
    #plt.show()
    plt.savefig(path + '/DecisionTree_test_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 DecisionTreeClassifier test
    clf.fit(X, y)
    prey = clf.predict(X_test)
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
    #plt.show()
    plt.savefig(path + '/DecisionTree_test_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str2 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

    # 画出ROC曲线 DecisionTreeClassifier train
    y_score = clf.fit(X, y).predict_proba(X_train)
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(path + '/DecisionTree_train_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 DecisionTree train
    clf.fit(X, y)
    prey = clf.predict(X_train)
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
    #plt.show()
    plt.savefig(path + '/DecisionTree_train_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str3 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

    import pickle
    pickle.dump(Classified_two_DecisionTree, open(path + "/Classified_two_DecisionTree.dat", "wb"))

    return str1, str2, str3

# 10.6
def SVM_classifier(a, b, c, d,e,path,csvName):
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
    import pickle

    data = pd.DataFrame(pd.read_csv(csvName))

    X = data.values[:, 1:-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    param_grid = {
        'C': [10, 70, 100],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    }

    # Create a SVM classifier object
    svc = SVC()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(svc, param_grid=param_grid, cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters found:", grid_search.best_params_)

    # Evaluate the best estimator on the test data
    print("Test accuracy:", grid_search.score(X_test, y_test))
    str1 = f"Best parameters: {grid_search.best_params_}" + "\n" + f"Test accuracy: {grid_search.score(X_test, y_test)}"

    if b==0.1:
        kernel1='linear'
    elif b==0.2:
        kernel1 = 'poly'
    elif b==0.3:
        kernel1 = 'rbf'
    else:
        kernel1 = 'sigmoid'

    if d==0.1:
        gamma1='scale'
    else:
        gamma1 = 'auto'


    if e==0:
        clf = SVC(degree=a, kernel=kernel1, C=c, gamma=gamma1, probability=True)
    clf.fit(X, y)
    Classified_two_SVM = clf.fit(X, y)

    svc_predictions = clf.predict(X_test)
    print("Accuracy of SVM using optimized parameters:", accuracy_score(y_test, svc_predictions) * 100)
    print("Report:", classification_report(y_test, svc_predictions))
    print("Score:", clf.score(X_test, y_test))
    str4 = "Accuracy of SVM using optimized parameters:" + str(accuracy_score(y_test, svc_predictions) * 100) + \
           '\n' + "Report:" + str(classification_report(y_test, svc_predictions)) + \
           "\n" + "Score:" + str(clf.score(X_test, y_test))


    # 画出ROC曲线 SVM test
    y_score = clf.fit(X, y).predict_proba(X_test)
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
    #plt.show()
    plt.savefig(path + '/SVM_test_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵 SVM
    clf.fit(X, y)
    prey = clf.predict(X_test)
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
    #plt.show()
    plt.savefig(path + '/SVM_test_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str2 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)

    # 画出ROC曲线 SVM train
    y_score = clf.fit(X, y).predict_proba(X_train)
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(path + '/SVM_train_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 画出混淆矩阵  SVM train
    clf.fit(X, y)
    prey = clf.predict(X_train)
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
    #plt.show()
    plt.savefig(path + '/SVM_train_CM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("true:", true)
    str3 = "fpr:" + str(fpr) + '\n' + "tpr:" + str(tpr) + "\n" + "true:" + str(true)
    import pickle
    pickle.dump(Classified_two_SVM, open(path + "/Classified_two_SVM.dat", "wb"))

    return str1, str4,str2, str3




def Visualization_for_classification(csvname,path,column_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 设置全局字体大小和样式
    sns.set(font_scale=3.0)
    sns.set_style("whitegrid")

    # 读取数据集
    data = pd.read_csv(csvname)  # 替换为你的数据集文件名

    # 按类别分割数据
    try:
        # 尝试访问data[column_name]
        value = data[column_name]
        class_0_data = data[data[column_name] == 0]  # 替换为你的类别列名
        class_1_data = data[data[column_name] == 1]  # 替换为你的类别列名

        # 可视化特征的分布
        for feature in data.columns[1:]:
            if feature != 'class':  # 排除类别列
                plt.figure(figsize=(10, 8))
                sns.histplot(class_0_data[feature], color='blue', label='Class 0', alpha=0.5, bins=20)
                sns.histplot(class_1_data[feature], color='red', label='Class 1', alpha=0.5, bins=20)
                plt.xlabel('Feature Value')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {feature}')
                plt.legend()
                plt.savefig(path + "/" + f'3_Distribution_of_{feature}.png', dpi=400)  # 保存图像，文件名为特征名
                plt.close()

        return True
    except KeyError:
        # 如果出现KeyError异常，说明data[column_name]不存在
        return False


def Symbolicregression_Modelconstruction(csvname,path):
    import pickle
    import matplotlib.pyplot as plot
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import svm
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from pandas import DataFrame
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    import pickle
    from gplearn.genetic import SymbolicRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from sklearn.tree import export_graphviz
    import pydotplus
    import graphviz
    from io import StringIO
    from IPython.display import Image


    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, :-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    feature_names = list(data.columns[:-1])

    # 定义符号回归模型，并使用训练数据拟合模型
    reg = SymbolicRegressor(population_size=5000, generations=5, verbose=1,
                            function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
                                          'inv', 'max', 'min', 'sin', 'cos', 'tan'],
                            metric='mean absolute error', stopping_criteria=0.001,
                            random_state=0)

    Symbolic_Regression_Model=reg.fit(X_train, y_train)
    import pickle
    pickle.dump(Symbolic_Regression_Model, open(path + "/Symbolic_Regression_Model.dat", "wb"))



    from sklearn.metrics import mean_absolute_error
    # 预测测试数据
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mae)



    from sklearn.model_selection import learning_curve

    # Compute the training and test scores
    train_sizes, train_scores, test_scores = learning_curve(
        reg, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_absolute_error')

    # Create the learning curve plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Testing score')
    plt.xlabel('Training examples')
    plt.ylabel('Score (MAE)')
    plt.ylim((-1, 1))
    plt.legend(loc='best')
    plt.savefig(path + '/Learning curve.png', dpi=300, bbox_inches='tight')
    plt.close()




    # 绘制预测值和真实值的散点图
    fig = plt.figure(dpi=300)
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    # 绘制一条参考线，x=y，表示预测值等于真实值的情况
    plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.ylim()[0], plt.ylim()[1]], ls="--", c=".3")
    plt.savefig(path + '/figure.png', dpi=300, bbox_inches='tight')
    plt.close()

    import sympy as sp
    import pydot

    expr = sp.simplify(str(reg._program))
    dot = sp.dotprint(expr, format='pydot')
    graph = pydot.graph_from_dot_data(dot)[0]
    for node in graph.get_nodes():
        if node.get_shape() == 'ellipse':  # 运算节点
            node.set_style('filled')
            node.set_fillcolor('#FFC0CB')  # 淡粉色
        else:  # 叶节点
            node.set_style('filled')
            node.set_fillcolor('#ADD8E6')  # 浅蓝色

    # 可视化符号表达式树
    graph.write_png(path + '/expression_tree.png')

    return mae

def Symbolicclassification(csvname,path):
    import pickle
    import matplotlib.pyplot as plot
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import svm
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from pandas import DataFrame
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    import pickle
    from gplearn.genetic import SymbolicClassifier
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from sklearn.tree import export_graphviz
    import pydotplus
    import graphviz
    from io import StringIO
    from IPython.display import Image

    data = pd.DataFrame(pd.read_csv(csvname))

    X = data.values[:, :-1]
    y = data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])

    # 创建符号分类器
    reg = SymbolicClassifier(population_size=5000, generations=30, tournament_size=20,
                             function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg',
                                           'inv', 'max', 'min', 'sin', 'cos', 'tan'],
                             stopping_criteria=0.0, const_range=(-1.0, 1.0), verbose=1)
    reg.fit(X_train, y_train)
    Symbolic_Classification_Model = reg.fit(X_train, y_train)
    import pickle
    pickle.dump(Symbolic_Classification_Model, open(path + "/Symbolic_Regression_Model.dat", "wb"))

    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    # Prediction on test set
    y_pred = reg.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(len(set(y))), set(y))
    plt.yticks(np.arange(len(set(y))), set(y))

    # Adding text to the confusion matrix cells with larger font size
    for i in range(len(set(y))):
        for j in range(len(set(y))):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=14)

    plt.savefig(path + '/test_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    # Prediction on test set
    y_pred = reg.predict(X_train)

    # Confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(np.arange(len(set(y))), set(y))
    plt.yticks(np.arange(len(set(y))), set(y))

    # Adding text to the confusion matrix cells with larger font size
    for i in range(len(set(y))):
        for j in range(len(set(y))):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=14)
    plt.savefig(path + '/train_confusion.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curve
    y_probs = reg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    # Plotting ROC Curve
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(path + '/test_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Accuracy
    accuracy = reg.score(X_test, y_test)
    str1="Test_ROC_Accuracy:"+str(accuracy)
    print("Accuracy:", accuracy)

    # ROC Curve
    y_probs = reg.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_probs)

    # Plotting ROC Curve
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.savefig(path + '/train_ROC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy
    accuracy = reg.score(X_train, y_train)
    str2 = "Train_ROC_Accuracy:" + str(accuracy)
    print("Accuracy:", accuracy)

    return str1,str2

def Result(csvname,path,model_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import ExtraTreesClassifier
    import pickle
    import numpy as np

    # 将 np.bool 替换为 np.bool_
    np.bool = np.bool_

    # 加载预测数据集
    data = pd.read_csv(csvname)

    import pickle  # 加载训练好的ExtraTreeClassifier模型

    model = pickle.load(open(model_path, "rb"))

    # 拟合模型
    #column_name=data.columns[-1]
    #X = data.drop(columns=[column_name])
    #y = data[column_name]

    X = data.drop(columns=['stability'])
    y = data['stability']


    # 初始化 SHAP explainer
    explainer = shap.Explainer(model, X)

    print(1)
    # 计算 SHAP 值
    shap_values = explainer(X)

    print(1)
    # 将 shap 值转换为 pandas DataFrame
    shap_df = pd.DataFrame(shap_values.values[:, :, 1], columns=X.columns)

    print(1)
    import matplotlib
    matplotlib.use('TkAgg')
    print(1)

    import matplotlib
    matplotlib.use('TkAgg')

    # 绘制蜂群图
    shap.summary_plot(shap_values.values[:, :, 1], X, show=False, plot_type='dot')
    plt.tight_layout()
    # 保存图表为 .png 格式的文件
    plt.savefig(path+'/summary_plot.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制第一个样本的活力图，绿色表示对目标分类的贡献，红色表示对其他分类的贡献
    shap.force_plot(explainer.expected_value[1], shap_values.values[0, :, 1], X.iloc[0, :],
                    matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(path+'/Forceplot.png', bbox_inches='tight', dpi=300)
    plt.close()

    import seaborn as sns

    # 计算每个特征的 SHAP 值绝对值的平均值
    shap_mean = np.abs(shap_df).mean()

    # 按平均 SHAP 值绝对值降序排列特征
    shap_mean_sorted = shap_mean.sort_values(ascending=False)

    # 绘制重要性排名柱状图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=shap_mean_sorted.values, y=shap_mean_sorted.index)
    plt.xlabel('Mean |SHAP| value', fontsize=13)
    plt.title('Feature Importance Rankings', fontsize=16)
    plt.savefig(path+'/Feature_ranking_bar.png', bbox_inches='tight', dpi=300)
    plt.close()

    ## 计算 SHAP 值
    import matplotlib.pyplot as plt
    # 将 SHAP 值转换为 Explanation 对象列表
    shap_values = [shap.Explanation(values=sv, base_values=np.mean(data, axis=0), data=data.iloc[[i]]) for i, sv in
                   enumerate(shap_values)]

    # 绘制每个样本的特征重要性瀑布图
    for i, sv in enumerate(shap_values):
        plt.title(f"Sample {i}")
        shap.waterfall_plot(sv[0], max_display=10)

        plt.savefig(path + '/Waterfall.png', bbox_inches='tight', dpi=300)
        plt.close()


    import graphviz
    from sklearn.tree import export_graphviz
    # 将决策树导出为DOT格式
    dot_data = export_graphviz(model.estimators_[0], out_file=None,
                               feature_names=data.columns[:-1], class_names=['0', '1'],
                               filled=True, rounded=True,
                               special_characters=True)
    # 将DOT格式转换为绘图
    graph = graphviz.Source(dot_data)

    # 展示决策树状图
    graph.render('decision_tree', format='png')
    graph.write_png(path + '/decision_tree.png')

















