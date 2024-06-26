from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from NJmatML.dataML import Symbolicregression_Modelconstruction, Symbolicclassification
from mainwindow import Ui_MainWindow
import dialog_Machinelearningmodeling_Algorithms
import dialog_Preprocessing_Featureranking,dialog_continuous_data_Xgboost,dialog_continuous_data_Random_Forest\
    ,dialog_continuous_data_Bagging,dialog_continuous_data_AdaBoost,dialog_continuous_data_GradientBoosting,\
    dialog_continuous_data_ExtraTree,dialog_continuous_data_Svm,dialog_continuous_data_DecisionTree,\
    dialog_continuous_data_LinearRegression,dialog_continuous_data_Ridge,dialog_continuous_data_MLP
import dialog_classified_data_two_RandomForest,dialog_classified_data_two_ExtraTree,dialog_classified_data_two_GaussianProcess,\
    dialog_classified_data_two_DecisionTree,dialog_classified_data_two_SVM


from NJmatML import dataML
import argparse
import warnings
#import rdkit
warnings.filterwarnings("ignore")

import os
import shutil
import numpy as np
import pandas

# untitled1.py 加 self.textBrowser.setLineWrapMode(0) 水平滑轮                   #Todo

class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("NJmatML")
        self.setWindowIcon(QtGui.QIcon("test.ico"))
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--origin_path_1", type=str,default="D:/project_zl/luanqibazao/2DEformationCleaned.csv",
                                 help="默认数据导入-训练测试集文件路径")
        self.parser.add_argument("--origin_path_2", type=str, default="D:/project_zl/luanqibazao/x_New.csv"
                                 ,help="预测集建立——导入预测集")
        self.parser.add_argument("--origin_path_3", type=str, default="D:/project_zl/Featurize_formula_exps.csv",
                                help="默认描述符生成-有机文件路径")
        self.parser.add_argument("--origin_path_4", type=str, default="D:/project_zl/Inorganic_formula.csv",
                                 help="默认描述符生成-无机文件路径")
        self.parser.add_argument("--origin_path_5", type=str, default="D:/project_zl/data_rfe25new.csv",
                                 help="没用")
        self.parser.add_argument("--origin_path_6", type=str,
                                 default="D:/project_zl/new_generate/Machine Learning Modeling/"
                                         "Algorithms/Continuous data/Xgboost/Continuous_Xgboost.dat",
                                 help="已生成模型导入-选择模型")
        self.parser.add_argument("--origin_path_7", type=str,
                                 default="D:/project_zl/new_generate/Shapley/Model import/.dat",
                                 help="shapley_Model import")
        self.parser.add_argument("--prediction_visualization_path", type=str, default="",
                                 help="Prediction visualization默认文件路径")
        #D:\project_zl\new_generate\Machine Learning Modeling\Algorithms\Continuous data\Xgboost
        self.parser.add_argument("--save_path", type=str, default="D:/new_generate",help="保存在此文件夹")
        self.parser.add_argument("--if_open", default=True, help="是否打开生成的文件(csv,png,txt,tif),true是打开")    # Todo 改
        self.parser.add_argument("--if_control", default=False, help="是否控制按键逻辑(1,2,3...),false是控制")           # Todo 改

        self.opt = self.parser.parse_args()                                # 封装参数

        self.num_1=0
        self.num_2 = 0
        self.num_3 = 0
        self.num_4 = 0
        self.num_5 = 0
        self.num_6 = 0
        self.num_7 = 0

        # 触发
        # 文件
        self.action.triggered.connect(self.file_directory)

        # 描述符生成
        self.action_smiles_2.triggered.connect(self.enter_organic_smiles)
        self.action_17.triggered.connect(self.enter_inorganic_fomula)
        self.actionpydel.triggered.connect(self.Descriptorgeneration_Organicmoleculardescriptors_pydel)
        self.actionrdkit_2.triggered.connect(self.Descriptorgeneration_Organicmoleculardescriptors_rdkit)
        self.actionmatminer_2.triggered.connect(self.Descriptorgeneration_Inorganicmoleculardescriptors_matminers)

        # 数据导入
        self.action_21.triggered.connect(self.enter_training_test_set_path)
        self.action_22.triggered.connect(self.Dataimporting_Trainingtestsetimportandvisualization_Importandvisualization)
        self.action_20.triggered.connect(self.Dataimporting_Heatmap)
        self.actionMore_visualization_only_for_classification.triggered.connect(self.Dataimporting_Morevisualization_classification)

        # 预处理
        self.actionrfe.triggered.connect(self.Preprocessing_Rfefeatureselection)
        self.action_rfe.triggered.connect(self.Preprocessing_Datavisualizationafterfeatureselection_Heatmap)
        self.action_rfe_pairplot.triggered.connect(self.Preprocessing_Datavisualizationafterfeatureselection_Pairplot)
        self.action_3.triggered.connect(self.show_dialog_Preprocessing_Featureranking_Featureimportancebeforefeatureselection)
        self.action_4.triggered.connect(self.show_dialog_Preprocessing_Featureranking_Featureimportanceafterfeatureselection)

        # 机器学习建模
        self.action_13.triggered.connect(self.show_dialog_Machinelearningmodeling_Algorithms)

        # 符号回归
        self.actionSymbolic_regression.triggered.connect(self.GP_Symbolicregression)
        self.actionsymbolic_classification.triggered.connect(self.GP_Symbolicclassification)

        # 预测集建立
        self.actionSelect_machine_learning_model.triggered.connect(self.import_model_dat)
        self.action_15.triggered.connect(self.Prediction_Importvirtualdata)

        self.action_18.triggered.connect(self.Prediction_Predictiongeneration)
        self.actionPrediction_visualization.triggered.connect(self.Prediction_Prediction_visualization)

        # shapley
        self.actionModel_import.triggered.connect(self.shapley_Modelimport)
        self.actionData_import_2.triggered.connect(self.shapley_Dataimport)
        self.actionResult.triggered.connect(self.shapley_Result)



        self.enter_organic_smiles_state =self.opt.if_control
        self.enter_inorganic_fomula_state = self.opt.if_control
        self.enter_training_test_set_path_state=self.opt.if_control                      # 由于本电脑有默认导入文件路径，所以可由if_control控制，方便测试


        self.training_test_set_path_state=self.opt.if_control                            # 控制按钮触发前后顺序
        self.heatmap_state = self.opt.if_control
        self.rfe_feature_selection_state = self.opt.if_control
        self.import_model_dat_state=self.opt.if_control
        self.import_prediction_dataset_state=self.opt.if_control
        self.import_prediction_dataset_state = self.opt.if_control


    # 状态重置
    def clear_state(self):
        self.training_test_set_path_state = self.opt.if_control
        self.heatmap_state = self.opt.if_control
        self.rfe_feature_selection_state = self.opt.if_control
        self.enter_training_test_set_path_state = self.opt.if_control


    def clear_state_Prediction(self):
        self.import_model_dat_state =False
        self.import_prediction_dataset_state =False
        self.prediction_generation=False

    def textBrowser_print_Pandas_DataFrame_table(self,data,if_columns,if_index):  # if_index,if_columns是否导入横竖坐标名称
        list2 = list(data.values)  # 坐标值

        if if_columns==0:                                                     # 无纵坐标名称
            pass
        elif if_columns==1:                                                   # 纵坐标名称默认为0,1...
            str1 = " " * 24
            for i in range(len(list2[0])):
                str_temp = "{str:^{len}}".format(str=str(i), len=24)
                str1 = str1 + str_temp
            self.textBrowser.append(str1)
        elif if_columns==2:
            list1 = list(data.columns)                                        # 纵坐标名称
            str1 = " "*24
            for i in range(len(list1)):
                str_temp = "{str:^{len}}".format(str=str(list1[i]), len=24)
                str1 = str1 + str_temp
            self.textBrowser.append(str1)

        if if_index==0:                                                   # 无横坐标名称
            pass
        elif if_index == 1:
            for i in range(len(list2)):
                str2 = "{str:^{len}}".format(str=str(i), len=24)          # 横坐标名称默认为0,1...
                for j in range(len(list2[0])):
                    str_temp = "{str:^{len}}".format(str=str(list2[i][j]), len=24)
                    str2 = str2 + str_temp
                self.textBrowser.append(str2)
        elif if_index==2:
            list3 = list(data.index)                                       # 横坐标名称
            for i in range(len(list2)):
                str2 = "{str:^{len}}".format(str=str(list3[i]), len=24)
                for j in range(len(list2[0])):
                    str_temp = "{str:^{len}}".format(str=str(list2[i][j]), len=24)
                    str2 = str2 + str_temp
                self.textBrowser.append(str2)

    def textBrowser_print_six(self,str1,scores,str2):
        self.textBrowser.append(str1)
        str3 = "scores: "
        str4 = ""
        for i in range(len(scores)):
            str3 = str3 + str(scores[i]) + "  "
            scores_mean = scores[:i + 1].mean()
            str4 = str4 + str(i + 1) + " " + "scores_mean: " + str(scores_mean) + "\n"
        self.textBrowser.append(str3)
        self.textBrowser.append(str4)
        self.textBrowser.append(str2)


    # 文件——保存路径-----------------------------------------------------------------------------------------
    def file_directory(self):
        self.opt.save_path= QFileDialog.getExistingDirectory(self,"Select folder","")  # 起始路径
        if self.opt.save_path=="":
            QMessageBox.information(self, 'Hint', 'The save path is empty!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)


    # 描述符生成----------------------------------------------------------------------------------------------
    # 导入有机smlies
    def enter_organic_smiles(self):
            directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
            if len(directory_temp) > 0:
                str_root = str(directory_temp)
                f_csv = str_root.rfind('.csv')
                if f_csv != -1:                                                    # 判断是不是.csv
                    self.opt.origin_path_3=str((str_root.replace("\\", '/'))[2:-2])
                    self.enter_organic_smiles_state = True
                    QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                else:
                    QMessageBox.information(self, 'Hint', 'Not .csv file, please re-enter!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Please enter a file!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

    # 导入无机化学式
    def enter_inorganic_fomula(self):
            directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
            if len(directory_temp) > 0:
                str_root = str(directory_temp)
                f_csv = str_root.rfind('.csv')
                if f_csv != -1:                                           # 判断是不是.csv
                    self.opt.origin_path_4= str((str_root.replace("\\", '/'))[2:-2])
                    self.enter_inorganic_fomula_state = True
                    QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                else:
                    QMessageBox.information(self, 'Hint', 'Not .csv file, please re-enter!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Please enter a file!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

    # 暂时不行
    # 描述符生成--有机分子描述符--pydel描述符   descriptor generation_Organic molecular descriptors_pydel
    def Descriptorgeneration_Organicmoleculardescriptors_pydel(self):
        if self.enter_organic_smiles_state == True:
            path = self.opt.save_path + "/Descriptor generation/Organic molecular descriptors/pydel"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            data2=dataML.smiles_csv_pydel(self.opt.origin_path_3)
            data2b=dataML.pydel_featurizer(path)
            self.textBrowser_print_Pandas_DataFrame_table(data2, 0, 1)
            self.textBrowser_print_Pandas_DataFrame_table(data2b, 2, 1)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path+"/pydel_featurizer_output.csv").replace("/", "\\")
                os.startfile(str1)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import smiles"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 描述符生成--有机描述符--rdkit描述符     descriptor generation_Organic molecular descriptors_rdkit
    def Descriptorgeneration_Organicmoleculardescriptors_rdkit(self):
        if self.enter_organic_smiles_state == True:
            path = self.opt.save_path + "/Descriptor generation/Organic molecular descriptors/rdkit"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            data3=dataML.smiles_csv_rdkit(self.opt.origin_path_3)
            data5=dataML.rdkit_featurizer(path)
            self.textBrowser_print_Pandas_DataFrame_table(data3,0,1)
            self.textBrowser_print_Pandas_DataFrame_table(data5, 2, 1)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path+"/rdkit_featurizer_output.csv").replace("/", "\\")
                os.startfile(str1)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import smiles"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 描述符生成--无机描述符--matminer无机材料描述符生成     descriptor generation_Inorganic molecular descriptors_matminer
    def Descriptorgeneration_Inorganicmoleculardescriptors_matminers(self):
        if self.enter_inorganic_fomula_state == True:
            path = self.opt.save_path + "/Descriptor generation/Inorganic molecular descriptors/matminer"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            data4=dataML.inorganic_csv(self.opt.origin_path_4)
            data8,element_fraction_labels=dataML.inorganic_featurizer(path)
            self.textBrowser_print_Pandas_DataFrame_table(data4, 2, 1)
            self.textBrowser_print_Pandas_DataFrame_table(data8, 2, 1)
            str1=""
            for i in range(len(element_fraction_labels)):
                str1=str1+str(element_fraction_labels[i])+" "
            self.textBrowser.append(str1)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path+"/inorganic_featurizer_output.csv").replace("/", "\\")
                os.startfile(str1)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import formula"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 数据导入----------------------------------------------------------------------------------------
    # 数据测试集路径
    def enter_training_test_set_path(self):
            directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
            if len(directory_temp)>0:
                str_root = str(directory_temp)
                f_csv = str_root.rfind('.csv')
                if f_csv!=-1:                                                 # 判断是不是.csv
                    self.opt.origin_path_1 = str((str_root.replace("\\",'/'))[2:-2])
                    self.clear_state()
                    self.enter_training_test_set_path_state = True
                    QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                else:
                    QMessageBox.information(self, 'Hint', 'Not .csv file, please re-enter!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Please enter a file!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

    # 数据导入--训练测试集导入与可视化--导入并可视化    Data importing_Training test set import and visualization_Import and visualization
    def Dataimporting_Trainingtestsetimportandvisualization_Importandvisualization(self):
        try:
            if self.enter_training_test_set_path_state == True:
                self.textBrowser.clear()

                self.training_test_set_path_state=True
                path=self.opt.save_path+"/Data importing/Training test set import and visualization/Import and visualization"
                if os.path.exists(path):                                       # 重做得删文件夹
                    shutil.rmtree(path)
                os.makedirs(path)
                data=dataML.file_name(self.opt.origin_path_1,path)                  #打开csv并存到data中
                dataML.hist(path)  #画所有列分布的柱状图，例如potential 在0.3 V最多

                # textBrowser写入
                self.textBrowser_print_Pandas_DataFrame_table(data,2,1)
                self.textBrowser.append("*"*150)

                QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

                if self.opt.if_open==True:
                    str1= (path+'/hist_allFeatures.png').replace("/", "\\")
                    os.startfile(str1)
                    str2 = (path + "/data.csv").replace("/", "\\")
                    os.startfile(str2)
            else:
                QMessageBox.information(self, 'Hint', 'Do "Path"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except Exception as e:
            print(e)

    #  数据导入-热图    Data importing_Heatmap
    def Dataimporting_Heatmap(self):
        if self.training_test_set_path_state==True:
            self.heatmap_state=True
            path = self.opt.save_path + "/Data importing/Heatmap"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            dataML.heatmap_before(path)  # 画封装函数特征选择之前heatmap热图
            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path+'/heatmap-before.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/heatmap-before.csv').replace("/", "\\")
                os.startfile(str2)

        else:
            QMessageBox.information(self, 'Hint', 'Do "Import and visualize"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    #  数据导入-更多可视化    Data importing_More visualization (only for classification)
    def Dataimporting_Morevisualization_classification(self):
        try:
            if self.enter_training_test_set_path_state == True:

                path = self.opt.save_path + "/Data importing/More visualization (only for classification)"
                csvname = self.opt.origin_path_1
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)

                value, ok = QtWidgets.QInputDialog.getText(self, "Train/Test dataset",
                                                           "type column name for conversion:",
                                                           QtWidgets.QLineEdit.Normal, "")
                result = dataML.Visualization_for_classification(csvname, path, value)
                if result == False:
                    QMessageBox.information(self, 'Hint', 'Column name not found!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                else:
                    if self.opt.if_open == True:
                        os.startfile(path)


            else:

                QMessageBox.information(self, 'Hint', 'Do "Path"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except Exception as e:
            print(e)

    # 预处理---------------------------------------------------------------------------------------------
    # 预处理-rfe特征选择           Preprocessing_rfe feature selection
    def Preprocessing_Rfefeatureselection(self):
        # 后面四个数字的作用依次是 初始值 最小值 最大值 步幅
        if self.training_test_set_path_state==True:

            value, ok = QtWidgets.QInputDialog.getInt(self, "RFE feature selection",
                                                      "Enter the number of features you want to have left:", 37, -10000, 10000, 2)

            # ok为true，则表示用户单击了OK（确定）按钮，若ok为false，则表示用户单击了Cancel（取消）按钮
            if ok:
                path = self.opt.save_path + "/Preprocessing/Rfe feature selection"
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)
                self.rfe_feature_selection_state = True
                target, data_rfe = dataML.feature_rfe_select1(value, path)
                with open(path+"/data.txt", "r") as f:                                   # 从txt写入textBrowser
                    while True:
                        line = f.readline()  # 包括换行符
                        line = line[:-1]  # 去掉换行符
                        if line:
                            self.textBrowser.append(str(line))
                        else:
                            break
                #self.textBrowser.append("目标target:")
                self.textBrowser.append("target:")
                self.textBrowser_print_Pandas_DataFrame_table(target,2,1)
                #self.textBrowser.append("rfe后的总数据data_rfe:")
                self.textBrowser.append("data_rfe(Total data after rfe):")
                self.textBrowser_print_Pandas_DataFrame_table(data_rfe,2,1)
                self.textBrowser.append("*" * 150)

                QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

                if self.opt.if_open == True:
                    str1 = (path + "/data.txt").replace("/", "\\")
                    os.startfile(str1)
                    str2 = (path + "/target.csv").replace("/", "\\")
                    os.startfile(str2)
                    str3 = (path + "/data_rfe.csv").replace("/", "\\")
                    os.startfile(str3)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import and visualize"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 预处理-特征选择后数据可视化-画rfe特征选择后的热图         Preprocessing_Data visualization after feature selection_Heat map
    def Preprocessing_Datavisualizationafterfeatureselection_Heatmap(self):
        if self.training_test_set_path_state == True:
            if self.rfe_feature_selection_state == True:
                path = self.opt.save_path + "/Preprocessing/Data visualization after feature selection/Heat map"
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)
                dataML.heatmap_afterRFE(path)
                QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
                if self.opt.if_open == True:
                    str1 = (path+'/heatmap-afterRFE.csv').replace("/", "\\")
                    os.startfile(str1)
                    str2 = (path + '/heatmap-afterRFE.png').replace("/", "\\")
                    os.startfile(str2)
            else:
                QMessageBox.information(self, 'Hint', 'Do "RFE feature selection"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import and visualize"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 预处理-特征选择后数据可视化-画rfe特征选择后的pairplot图         Preprocessing_Data visualization after feature selection_Pairplot
    def Preprocessing_Datavisualizationafterfeatureselection_Pairplot(self):
        if self.training_test_set_path_state == True:
            if self.rfe_feature_selection_state == True:
                path = self.opt.save_path + "/Preprocessing/Data visualization after feature selection/Pairplot"
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)
                dataML.pairplot_afterRFE(path)
                QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
                if self.opt.if_open == True:
                    str1 = (path+'/sns-pairplot-remain.png').replace("/", "\\")
                    os.startfile(str1)
            else:
                QMessageBox.information(self, 'Hint', 'Do "RFE feature selection"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import and visualize"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 预处理-重要性排名（皮尔逊系数）-特征选择之前的特征重要性               Preprocessing_Importance_ranking(Pearson)_Feature importance before feature selection
    def Preprocessing_Featureranking_Featureimportancebeforefeatureselection(self):

        path = self.opt.save_path + "/Preprocessing/Importance_ranking(Pearson)/Feature importance before feature selection"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.di.close()
        importance=dataML.FeatureImportance_before(int(self.num_1),int(self.num_2),int(self.num_3),int(self.num_4),path)  # 输入一定要int
                            # 用完就初始化，作为全局得共用

        self.textBrowser_print_Pandas_DataFrame_table(importance,2,2)
        self.textBrowser.append("*" * 150)

        QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                QMessageBox.Close)
        if self.opt.if_open == True:
            str1 = (path+'/FeatureImportance_before.png').replace("/", "\\")
            os.startfile(str1)

    def show_dialog_Preprocessing_Featureranking_Featureimportancebeforefeatureselection(self):
        def give(a,b,c,d):
            self.num_1=a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d

            self.Preprocessing_Featureranking_Featureimportancebeforefeatureselection()
        if self.training_test_set_path_state == True:
            if self.heatmap_state == True:
                if self.rfe_feature_selection_state == True:
                    self.di = QtWidgets.QDialog()
                    d = dialog_Preprocessing_Featureranking.Ui_Dialog()
                    d.setupUi(self.di)
                    self.di.show()
                    d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()))
                    d.buttonBox.rejected.connect(self.di.close)

                else:
                    QMessageBox.information(self, 'Hint', 'Do "RFE feature selection"!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Do "Heat map"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import and visualize"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 预处理-重要性排名（皮尔逊系数）-特征选择之后的特征重要性               Preprocessing_Importance_ranking(Pearson)_Feature importance after feature selection
    def Preprocessing_Featureranking_Featureimportanceafterfeatureselection(self):

        path = self.opt.save_path + "/Preprocessing/Importance_ranking(Pearson)/Feature importance after feature selection"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.di.close()
        importance_rfe=dataML.FeatureImportance_afterRFE(int(self.num_1),int(self.num_2),int(self.num_3),int(self.num_4),path)


        self.textBrowser_print_Pandas_DataFrame_table(importance_rfe, 2, 2)
        self.textBrowser.append("*" * 150)

        QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                QMessageBox.Close)
        if self.opt.if_open == True:
            str1 = (path+'/FeatureImportance_after.png').replace("/", "\\")
            os.startfile(str1)

    def show_dialog_Preprocessing_Featureranking_Featureimportanceafterfeatureselection(self):
        def give(a,b,c,d):
            self.num_1=a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d
            self.Preprocessing_Featureranking_Featureimportanceafterfeatureselection()

        if self.training_test_set_path_state == True:
            if self.heatmap_state == True:
                if self.rfe_feature_selection_state == True:
                    self.di = QtWidgets.QDialog()
                    d = dialog_Preprocessing_Featureranking.Ui_Dialog()
                    d.setupUi(self.di)
                    self.di.show()

                    d.buttonBox.accepted.connect(
                        lambda: give(d.lineEdit.text(), d.lineEdit_2.text(), d.lineEdit_3.text(), d.lineEdit_4.text()))
                    d.buttonBox.rejected.connect(self.di.close)
                else:
                    QMessageBox.information(self, 'Hint', 'Do "RFE feature selection"!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Do "Heat map"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import and visualize"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 机器学习建模--------------------------------------------------------------------------------------------------
    # 机器学习建模-模型选择                               Machine Learning Modeling_Algorithms
    def show_dialog_Machinelearningmodeling_Algorithms(self):
        #["Xgboost", "Random Forest","Bagging","AdaBoost","Gradient Boosting","Extra Tree"
    #,"Svm","Decision Tree","Linear Regression","Ridge","MLP"]
    #["Random Forest","Extra Tree","Gaussian Process","KNeighbors","Decision Tree","SVM"]
        def select(text1,text2):
            if text1=="Regression":
                if text2=="Xgboost":
                    self.show_dialog_continuous_data_Xgboost()
                elif text2=="Random Forest":
                    self.show_dialog_continuous_data_Random_Forest()
                elif text2=="Bagging":
                    self.show_dialog_continuous_data_Bagging()
                elif text2 == "AdaBoost":
                    self.show_dialog_continuous_data_AdaBoost()
                elif text2 == "Gradient Boosting":
                    self.show_dialog_continuous_data_GradientBoosting()
                elif text2 == "Extra Tree":
                    self.show_dialog_continuous_data_ExtraTree()
                elif text2 == "Svm":
                    self.show_dialog_continuous_data_Svm()
                elif text2 == "Decision Tree":
                    self.show_dialog_continuous_data_DecisionTree()
                elif text2 == "Linear Regression":
                    self.show_dialog_continuous_data_LinearRegression()
                elif text2 == "Ridge":
                    self.show_dialog_continuous_data_Ridge()
                elif text2 == "MLP":
                    self.show_dialog_continuous_data_MLP()
            else:
                if text2=="Random Forest":
                    self.show_dialog_classified_data_two_RandomForest()
                elif text2=="Extra Tree":
                    self.show_dialog_classified_data_two_ExtraTree()
                elif text2 == "Gaussian Process":
                    self.show_dialog_classified_data_two_GaussianProcess()
                elif text2 == "KNeighbors":
                    pass
                elif text2 == "Decision Tree":
                    self.show_dialog_classified_data_two_DecisionTree()
                elif text2 == "SVM":
                    self.show_dialog_classified_data_two_SVM()
        if self.training_test_set_path_state == True:
            if self.rfe_feature_selection_state == True:
                self.di = QtWidgets.QDialog()
                d = dialog_Machinelearningmodeling_Algorithms.Ui_Dialog()
                d.setupUi(self.di)
                self.di.show()
                d.buttonBox.accepted.connect(lambda: select(d.comboBox.currentText(), d.comboBox_2.currentText()))
                d.buttonBox.rejected.connect(self.di.close)
            else:
                QMessageBox.information(self, 'Hint', 'Do "RFE feature selection"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import and visualize"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 机器学习建模-模型选择-连续型-Xgboost                         Continuous data_Xgboost
    def Continuousdata_Xgboost(self):
        path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Xgboost"
        csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.di.close()

        str1, scores, str2 =dataML.xgboost_modify(int(self.num_1),int(self.num_2),self.num_3,self.num_4
                              ,self.num_5,self.num_6,self.num_7,path,csvname)  # 三个图，第一个图测试集拟合，第二个图交叉验证，第三个图训练集的拟合（没什么用）

        # (n_estimators=1000, max_depth=200, eta=0.2, gamma=0, subsample=0.9, colsample_bytree=0.8, learning_rate=0.2)
        self.textBrowser_print_six(str1, scores, str2)
        self.textBrowser.append("*" * 150)

        QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                QMessageBox.Close)
        if self.opt.if_open == True:
            str1 = (path+'/xgboost-modify.png').replace("/", "\\")
            os.startfile(str1)
            str2 = (path+'/xgboost_modify-10-fold-crossvalidation.png').replace("/", "\\")
            os.startfile(str2)
            str3 = (path+'/xgboost-modify-train-default.png').replace("/", "\\")
            os.startfile(str3)
    def show_dialog_continuous_data_Xgboost(self):
        def give(a,b,c,d,e,f,g):
            self.num_1=a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d
            self.num_5 = e
            self.num_6 = f
            self.num_7 = g
            self.Continuousdata_Xgboost()



        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_Xgboost.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                  ,d.lineEdit_5.text(),d.lineEdit_6.text(),d.lineEdit_7.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-Random Forest                         Continuous data_Random Forest
    def Continuousdata_RandomForest(self):

            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Random Forest"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()
            try:
                str1, scores, str2 = dataML.RandomForest_modify(int(self.num_1),float(self.num_2),int(self.num_3),int(self.num_4)
                                  ,int(self.num_5),path,csvname)
                self.textBrowser_print_six(str1, scores, str2)
                self.textBrowser.append("*" * 150)

            except Exception as e:
                print(e)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path+'/RandomForest-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path+'/RandomForest_modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path+'/RandomForest-modify-train.png').replace("/", "\\")
                os.startfile(str3)
    def show_dialog_continuous_data_Random_Forest(self):
        def give(a,b,c,d,e):
            self.num_1=a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d
            self.num_5 = e
            self.Continuousdata_RandomForest()


        self.six_state = True
        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_Random_Forest.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit_2.text(),d.lineEdit.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                  ,d.lineEdit_5.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-Bagging                         Continuous data_Bagging
    def Continuousdata_Bagging(self):
        path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Bagging"
        csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self.di.close()

        str1, scores, str2 = dataML.Bagging_modify(int(self.num_1), float(self.num_2), float(self.num_3), path,
                                                    csvname)  # 三个图，第一个图测试集拟合，第二个图交叉验证，第三个图训练集的拟合（没什么用）

        # (n_estimators=10, max_samples=1,max_features=1)
        self.textBrowser_print_six(str1, scores, str2)
        self.textBrowser.append("*" * 150)

        QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                QMessageBox.Close)
        if self.opt.if_open == True:
            str1 = (path + '/Bagging-modify.png').replace("/", "\\")
            os.startfile(str1)
            str2 = (path + '/Bagging-modify-10-fold-crossvalidation.png').replace("/", "\\")
            os.startfile(str2)
            str3 = (path + '/Bagging-modify-train.png').replace("/", "\\")
            os.startfile(str3)
    def show_dialog_continuous_data_Bagging(self):
        def give(a, b, c):
            self.num_1 = a
            self.num_2 = b
            self.num_3 = c

            self.Continuousdata_Bagging()

        try:
            self.di = QtWidgets.QDialog()
            d = dialog_continuous_data_Bagging.Ui_Dialog()
            d.setupUi(self.di)
            self.di.show()

            d.buttonBox.accepted.connect(
                lambda: give(d.lineEdit.text(), d.lineEdit_2.text(), d.lineEdit_3.text()))
            d.buttonBox.rejected.connect(self.di.close)
        except Exception as e:
            print(e)

    # 机器学习建模-模型选择-连续型-AdaBoost                         Continuous data_AdaBoost
    def Continuousdata_AdaBoost(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/AdaBoost"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1, scores, str2 = dataML.AdaBoost_modify(int(self.num_1), float(self.num_2), float(self.num_3),path,csvname)

            # (n_estimators=50, learning_rate=1, loss='linear')
            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/AdaBoost-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/AdaBoost-modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/AdaBoost-modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
            print(e)
    def show_dialog_continuous_data_AdaBoost(self):
        def give(a, b, c):
            self.num_1 = a
            self.num_2 = b
            if c[1:-1] == "linear" or c[:] == "linear":  # ['linear','square','exponential']
                self.num_3 = 0.1
            elif c[1:-1] == "square" or c[:] == "square":
                self.num_3 = 0.2
            elif c[1:-1] == "exponential" or c[:] == "exponential":
                self.num_3 = 0.3

            self.Continuousdata_AdaBoost()

        try:
            self.di = QtWidgets.QDialog()
            d = dialog_continuous_data_AdaBoost.Ui_Dialog()
            d.setupUi(self.di)
            self.di.show()

            d.buttonBox.accepted.connect(
                lambda: give(d.lineEdit.text(), d.lineEdit_2.text(), d.lineEdit_3.text()))
            d.buttonBox.rejected.connect(self.di.close)
        except Exception as e:
            print(e)

    # 机器学习建模-模型选择-连续型-Gradient Boosting                        Continuous data_Gradient Boosting
    def Continuousdata_GradientBoosting(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Gradient Boosting"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1, scores, str2 = dataML.GradientBoosting_modify(int(self.num_1), int(self.num_2), float(self.num_3),
                                                                int(self.num_4), float(self.num_5), path,
                                                                csvname)  # 三个图，第一个图测试集拟合，第二个图交叉验证，第三个图训练集的拟合（没什么用）

            # (n_estimators': 100, 'max_depth': 3, 'min_samples_split': 1,'min_samples_leaf': 1,'learning_rate': 0.1)
            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/GradientBoosting-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/GradientBoosting-modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/GradientBoosting-modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
            print(e)
    def show_dialog_continuous_data_GradientBoosting(self):
        def give(a, b, c, d, e):
            self.num_1 = a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d
            self.num_5 = e

            self.Continuousdata_GradientBoosting()

        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_GradientBoosting.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(
            lambda: give(d.lineEdit.text(), d.lineEdit_2.text(), d.lineEdit_3.text(), d.lineEdit_4.text(),
                         d.lineEdit_7.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-Extra Tree                         Continuous data_Extra Tree
    def Continuousdata_ExtraTree(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Extra Tree"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()
            # max_depth=None,max_features['sqrt', 'log2', None,'auto']='auto',min_samples_split=2,n_estimators=100,random_state=None
            str1, scores, str2 = dataML.ExtraTree_modify(int(self.num_1), float(self.num_2), int(self.num_3),
                                                                float(self.num_5), path,
                                                                csvname)  # 三个图，第一个图测试集拟合，第二个图交叉验证，第三个图训练集的拟合（没什么用）

            # (n_estimators': 100, 'max_depth': 3, 'min_samples_split': 2,'min_samples_leaf': 1,'learning_rate': 0.1)
            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/ExtraTree-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/ExtraTree-modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/ExtraTree-modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
            print(e)
    def show_dialog_continuous_data_ExtraTree(self):
        # max_depth=None,max_features['sqrt', 'log2', None,'auto']='auto',min_samples_split=2,random_state=None
        def give(a, b, c,e):
            if a == 'None':
                self.num_1 = 0
            else:
                self.num_1 = a
            if b[1:-1] == "sqrt" or b[:] == "sqrt":
                self.num_2 = 0.1
            elif b[1:-1] == "log2" or b[:] == "log2":
                self.num_2 = 0.2
            elif b == 'None':
                self.num_2 = 0
            elif b[1:-1] == "auto" or b[:] == "auto":
                self.num_2 = 0.3
            else:
                self.num_2 = int(b)
            self.num_3 = c
            if e == 'None':
                self.num_5 = 0
            else:
                self.num_5 = e

            self.Continuousdata_ExtraTree()

        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_ExtraTree.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()
        d.buttonBox.accepted.connect(
            lambda: give(d.lineEdit_2.text(), d.lineEdit.text(), d.lineEdit_3.text(),
                         d.lineEdit_5.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-Svm                        Continuous data_Svm
    def Continuousdata_Svm(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Svm"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()
            # C=1.0, epsilon=0.1
            str1, scores, str2 = dataML.Svm_modify(float(self.num_1), float(self.num_2), path, csvname)

            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/Svm-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/Svm-modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/Svm-modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
            print(e)
    def show_dialog_continuous_data_Svm(self):
        def give(a, b):
            self.num_1 = a
            self.num_2 = b
            self.Continuousdata_Svm()

        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_Svm.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(
            lambda: give(d.lineEdit_2.text(), d.lineEdit.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-Decision Tree                        Continuous data_Decision Tree
    def Continuousdata_DecisionTree(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Decision Tree"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1, scores, str2 = dataML.DecisionTree_modify(int(self.num_1), float(self.num_2), int(self.num_3),
                                                            int(self.num_4), int(self.num_5), int(self.num_6), path,
                                                            csvname)
            # max_depth=None,max_features=None,min_samples_split=2,min_samples_leaf=1,random_state=None,max_leaf_nodes=None
            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/DecisionTree-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/DecisionTree-modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/DecisionTree-modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
            print(e)
    def show_dialog_continuous_data_DecisionTree(self):
        # max_depth=None,max_features=None,min_samples_split=2,min_samples_leaf=1,random_state=None,max_leaf_nodes=None
        def give(a, b, c, d, e, f):
            if a == 'None':
                self.num_1 = 0
            else:
                self.num_1 = a
            if b[1:-1] == "sqrt" or b[:] == "sqrt":  # 有整数，小数，None,'sqrt','log2'
                self.num_2 = 0.1
            elif b[1:-1] == "log2" or b[:] == "log2":
                self.num_2 = 0.2
            elif b == 'None':
                self.num_2 = 0
            elif b[1:-1] == "auto" or b[:] == "auto":
                self.num_2 = 0.3
            else:
                self.num_2 = int(e)
            self.num_3 = c
            self.num_4 = d
            if e == 'None':
                self.num_5 = 0
            else:
                self.num_5 = e
            if f == 'None':
                self.num_6 = 0
            else:
                self.num_6 = f
            self.Continuousdata_DecisionTree()

        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_DecisionTree.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(
            lambda: give(d.lineEdit_2.text(), d.lineEdit.text(), d.lineEdit_3.text(), d.lineEdit_4.text()
                         , d.lineEdit_5.text(), d.lineEdit_6.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-Linear Regression                         Continuous data_Linear Regression
    def Continuousdata_LinearRegression(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Linear Regression"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1, scores, str2 = dataML.LinearRegression_modify(int(self.num_1), int(self.num_2), int(self.num_3),
                                                                int(self.num_4), path, csvname)
            # fit_intercept=True, normalize=False, copy_X=True, n_jobs=None
            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/LinearRegression-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/LinearRegression-modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/LinearRegression-modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
            print(e)
    def show_dialog_continuous_data_LinearRegression(self):
        # fit_intercept=True, normalize=False, copy_X=True, n_jobs=None
        def give(a, b, c, d):
            if a == 'False':
                self.num_1 = 0
            else:
                self.num_1 = 1
            if b == 'False':
                self.num_2 = 0
            else:
                self.num_2 = 1
            if c == 'False':
                self.num_3 = 0
            else:
                self.num_3 = 1
            if d == 'None':
                self.num_4 = 0
            else:
                self.num_4 = d

            self.Continuousdata_LinearRegression()

        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_LinearRegression.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(
            lambda: give(d.lineEdit.text(), d.lineEdit_2.text(), d.lineEdit_3.text(), d.lineEdit_4.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-Ridge                         Continuous data_Ridge
    def Continuousdata_Ridge(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/Ridge"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1, scores, str2 = dataML.Ridge_modify(float(self.num_1), int(self.num_2), int(self.num_3),
                                                     int(self.num_4), int(self.num_5), path, csvname)
            # alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, random_state=None
            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/Ridge-modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/Ridge-modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/Ridge-modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
            print(e)
    def show_dialog_continuous_data_Ridge(self):
        # alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, random_state=None
        def give(a, b, c, d, e):
            self.num_1 = a
            if b == 'False':
                self.num_2 = 0
            else:
                self.num_2 = 1
            if c == 'False':
                self.num_3 = 0
            else:
                self.num_3 = 1
            if d == 'False':
                self.num_4 = 0
            else:
                self.num_4 = 1
            if e == 'None':
                self.num_5 = 0
            else:
                self.num_5 = e
            self.Continuousdata_Ridge()

        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_Ridge.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(
            lambda: give(d.lineEdit.text(), d.lineEdit_2.text(), d.lineEdit_3.text(), d.lineEdit_4.text(),
                         d.lineEdit_5.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-连续型-MLP                         Continuous data_MLP
    def Continuousdata_MLP(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Continuous data/MLP"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()


            str1, scores, str2 = dataML.MLP_modify(float(self.num_1),float(self.num_2),int(self.num_3),int(self.num_4)
                                ,int(self.num_5),path,csvname)
            self.textBrowser_print_six(str1, scores, str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path+'/MLP_modify.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path+'/MLP_modify-10-fold-crossvalidation.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path+'/MLP_modify-train.png').replace("/", "\\")
                os.startfile(str3)
        except Exception as e:
                print(e)
    def show_dialog_continuous_data_MLP(self):
        def give(a,b,c,d,e):
            self.num_1=a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d
            self.num_5 = e
            self.Continuousdata_MLP()


        self.di = QtWidgets.QDialog()
        d = dialog_continuous_data_MLP.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                          ,d.lineEdit_5.text()))
        d.buttonBox.rejected.connect(self.di.close)



    # 机器学习建模-模型选择-二分类-Random Forest                         Classified data_Two_Random Forest
    def Classifieddata_Two_RandomForest(self):
        try:

            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Classified data(two)/Random Forest"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1,str2,str3=dataML.randomforest_Classifier(int(self.num_1),int(self.num_2),int(self.num_3),int(self.num_4),
                                               int(self.num_5),int(self.num_6), path,csvname)

            # (max_depth=7, random_state=0, min_samples_leaf=1, max_features=1, min_samples_split=2,n_estimators=100)
            self.textBrowser.append(str1)
            self.textBrowser.append(str2)
            self.textBrowser.append(str3)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/RandomForest_test_ROC.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/RandomForest_test_CM.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/RandomForest_train_ROC.png').replace("/", "\\")
                os.startfile(str3)
                str4 = (path + '/RandomForest_train_CM.png').replace("/", "\\")
                os.startfile(str4)

        except Exception as e:
            print(e)
    def show_dialog_classified_data_two_RandomForest(self):
        def give(a,b,c,d,e,f):
            self.num_1=a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d
            self.num_5 = e
            self.num_6 = f
            self.Classifieddata_Two_RandomForest()


        self.di = QtWidgets.QDialog()
        d = dialog_classified_data_two_RandomForest.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                      ,d.lineEdit_5.text(),d.lineEdit_6.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-二分类-Extra Tree                         Classified data_Two_Extra Tree
    def Classifieddata_Two_ExtraTree(self):
        try:

            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Classified data(two)/Extra Tree"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1,str2,str3=dataML.extratrees_classifier(int(self.num_1),int(self.num_2),int(self.num_3),int(self.num_4),
                                               float(self.num_5),int(self.num_6),path,csvname)


            # (n_estimators=2, max_depth=None, min_samples_split=2, random_state=1, max_features='sqrt',
            #                                min_samples_leaf=4)
            self.textBrowser.append(str1)
            self.textBrowser.append(str2)
            self.textBrowser.append(str3)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/ExtraTrees_test_ROC.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/ExtraTrees_test_CM.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/ExtraTrees_train_ROC.png').replace("/", "\\")
                os.startfile(str3)
                str4 = (path + '/ExtraTrees_train_CM.png').replace("/", "\\")
                os.startfile(str4)
        except Exception as e:
            print(e)
    def show_dialog_classified_data_two_ExtraTree(self):
        def give(a,b,c,d,e,f):
            self.num_1=a

            # 只能传数值，通过数值进入extratrees_classifier()判断超参数是None,'sqrt'...
            if b=='None':
                self.num_2= 0
            else:
                self.num_2 = b
            self.num_3 = c
            self.num_4 = d

            if e[1:-1]=="sqrt" or e[:]=="sqrt":                         #有整数，小数，None,'sqrt','log2'
                self.num_5= 0.1
            elif e[1:-1]=="log2" or e[:]=="log2":
                self.num_5 = 0.2
            elif e=='None':
                self.num_5 = 0
            else:
                self.num_5 = int(e)

            self.num_6 = f
            self.Classifieddata_Two_ExtraTree()


        self.di = QtWidgets.QDialog()
        d = dialog_classified_data_two_ExtraTree.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                      ,d.lineEdit_5.text(),d.lineEdit_6.text()))
        d.buttonBox.rejected.connect(self.di.close)


    # 机器学习建模-模型选择-二分类-Gaussian Process                         Classified data_Two_Gaussian Process
    def Classifieddata_Two_GaussianProcess(self):
        try:

            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Classified data(two)/Gaussian Process"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1,str2,str3=dataML.GaussianProcess_classifier(float(self.num_1),float(self.num_2),int(self.num_3),int(self.num_4),
                                               int(self.num_5), path,csvname)


            # (kernel=1 ** 2 * RBF(length_scale=2), max_iter_predict=5, n_restarts_optimizer=2,
            #                                     optimizer='fmin_l_bfgs_b')
            self.textBrowser.append(str1)
            self.textBrowser.append(str2)
            self.textBrowser.append(str3)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/GaussianProcess_test_ROC.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/GaussianProcess_test_CM.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/GaussianProcess_train_ROC.png').replace("/", "\\")
                os.startfile(str3)
                str4 = (path + '/GaussianProcess_train_CM.png').replace("/", "\\")
                os.startfile(str4)

        except Exception as e:
            print(e)
    def show_dialog_classified_data_two_GaussianProcess(self):
        def give(a,b,c,d,e):
            self.num_1=a
            self.num_2 = b
            self.num_3 = c
            self.num_4 = d

            if e[1:-1]=="fmin_l_bfgs_b" or e[:]=="fmin_l_bfgs_b":
                self.num_5=0
            else:
                QMessageBox.information(self, 'Hint', 'Optimizer cannot be changed!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

            self.Classifieddata_Two_GaussianProcess()



        self.di = QtWidgets.QDialog()
        d = dialog_classified_data_two_GaussianProcess.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                          ,d.lineEdit_5.text()))

        d.buttonBox.rejected.connect(self.di.close)


    # 机器学习建模-模型选择-二分类-KNeighbors                         Classified data_Two_Random Forest


    # 机器学习建模-模型选择-二分类-Decision Tree                         Classified data_Two_Decision Tree
    def Classifieddata_Two_DecisionTree(self):
        try:
            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Classified data(two)/Decision Tree"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1,str2,str3=dataML.DecisionTree_classifier(float(self.num_1),int(self.num_2),float(self.num_3),int(self.num_4),
                                               int(self.num_5),path,csvname)

            # (criterion='gini', max_depth=5, max_features='auto', min_samples_leaf=4,
            #                                  min_samples_split=2)
            self.textBrowser.append(str1)
            self.textBrowser.append(str2)
            self.textBrowser.append(str3)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/DecisionTree_test_ROC.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/DecisionTree_test_CM.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/DecisionTree_train_ROC.png').replace("/", "\\")
                os.startfile(str3)
                str4 = (path + '/DecisionTree_train_CM.png').replace("/", "\\")
                os.startfile(str4)

        except Exception as e:
            print(e)
    def show_dialog_classified_data_two_DecisionTree(self):
        def give(a,b,c,d,e):
            if a[1:-1] == "gini" or a[:] == "gini":  # 'gini', 'entropy'
                self.num_5 = 0.1
            elif a[1:-1] == "entropy" or a[:] == "entropy":
                self.num_5 = 0.2
            else:
                QMessageBox.information(self, 'Hint', 'Criterion parameter error!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

            if b=='None':
                self.num_2= 0
            else:
                self.num_2 = b
            if c[1:-1] == "auto" or a[:] == "auto":  #   'auto', 'sqrt', 'log2', None
                self.num_3 = 0.1
            elif c[1:-1] == "sqrt" or c[:] == "sqrt":
                self.num_3 = 0.2
            elif c[1:-1] == "log2" or c[:] == "log2":
                self.num_3 = 0.3
            elif c=='None':
                self.num_3 = 0
            else:
                QMessageBox.information(self, 'Hint', 'max_features parameter error!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
            self.num_4 = d
            self.num_5 = e
            self.Classifieddata_Two_DecisionTree()



        self.di = QtWidgets.QDialog()
        d = dialog_classified_data_two_DecisionTree.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                      ,d.lineEdit_5.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # 机器学习建模-模型选择-二分类-SVM                        Classified data_Two_SVM
    def Classifieddata_Two_SVM(self):
        try:

            path = self.opt.save_path + "/Machine Learning Modeling/Algorithms/Classified data(two)/SVM"
            csvname = self.opt.save_path + "/Preprocessing/Rfe feature selection" + "/data_rfe.csv"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            self.di.close()

            str1,str4,str2,str3=dataML.SVM_classifier(int(self.num_1),float(self.num_2),int(self.num_3),float(self.num_4),
                                               float(self.num_5),path,csvname)


            # (degree=2, kernel='sigmoid', C=10, gamma='scale', probability=True)
            self.textBrowser.append(str1)
            self.textBrowser.append(str4)
            self.textBrowser.append(str2)
            self.textBrowser.append(str3)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            if self.opt.if_open == True:
                str1 = (path + '/SVM_test_ROC.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/SVM_test_CM.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/SVM_train_ROC.png').replace("/", "\\")
                os.startfile(str3)
                str4 = (path + '/SVM_train_CM.png').replace("/", "\\")
                os.startfile(str4)

        except Exception as e:
            print(e)
    def show_dialog_classified_data_two_SVM(self):
        def give(a,b,c,d,e):
            self.num_1=a

            if b[1:-1]=="linear" or b[:]=="linear":                         # ['linear', 'poly', 'rbf', 'sigmoid']
                self.num_2= 0.1
            elif b[1:-1]=="poly" or b[:]=="poly":
                self.num_2 = 0.2
            elif b[1:-1]=="rbf" or b[:]=="rbf":
                self.num_2 = 0.3
            elif b[1:-1]=="sigmoid" or b[:]=="sigmoid":
                self.num_2 = 0.4
            else:
                QMessageBox.information(self, 'Hint', 'Kernel parameter error!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
            self.num_3 = c
            if d[1:-1]=="scale" or d[:]=="scale":                         # ['scale', 'auto']
                self.num_4= 0.1
            elif d[1:-1]=="auto" or d[:]=="auto":
                self.num_4 = 0.2
            else:
                QMessageBox.information(self, 'Hint', 'Gamma parameter error!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
            if e=='True':
                self.num_5 = 0
            else:
                QMessageBox.information(self, 'Hint', 'Probability parameter error!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

            self.Classifieddata_Two_SVM()


        self.di = QtWidgets.QDialog()
        d = dialog_classified_data_two_SVM.Ui_Dialog()
        d.setupUi(self.di)
        self.di.show()

        d.buttonBox.accepted.connect(lambda :give(d.lineEdit.text(),d.lineEdit_2.text(),d.lineEdit_3.text(),d.lineEdit_4.text()
                                                      ,d.lineEdit_5.text()))
        d.buttonBox.rejected.connect(self.di.close)

    # GP-------------------------------------------------------------------------------------------------------
    # GP——符号回归                              GP_Symbolic regression
    def GP_Symbolicregression(self):
        try:
                directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
                if len(directory_temp) > 0:
                    str_root = str(directory_temp)
                    f_csv = str_root.rfind('.csv')
                    if f_csv != -1:                                                    # 判断是不是.csv
                        csvname=str((str_root.replace("\\", '/'))[2:-2])
                        path = self.opt.save_path + "/GP/Symbolic regression"
                        if os.path.exists(path):
                            shutil.rmtree(path)
                        os.makedirs(path)

                        MAE=Symbolicregression_Modelconstruction(csvname,path)
                        string1 = "MAE: " + str(MAE)
                        self.textBrowser.append(string1)

                        if self.opt.if_open == True:
                            str1 = (path + '/Learning curve.png').replace("/", "\\")
                            os.startfile(str1)
                            str2 = (path + '/figure.png').replace("/", "\\")
                            os.startfile(str2)
                            str3 = (path + '/expression_tree.png').replace("/", "\\")
                            os.startfile(str3)

                        QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                                QMessageBox.Close)

                    else:
                        QMessageBox.information(self, 'Hint', 'Not .csv file, please re-enter!', QMessageBox.Ok | QMessageBox.Close,
                                                QMessageBox.Close)
                else:
                    QMessageBox.information(self, 'Hint', 'Please enter a file!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
        except Exception as e:
            print(e)

    # GP——符号分类                                   GP_Symbolic classification
    def GP_Symbolicclassification(self):
        try:
            directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
            if len(directory_temp) > 0:
                str_root = str(directory_temp)
                f_csv = str_root.rfind('.csv')
                if f_csv != -1:  # 判断是不是.csv
                    csvname = str((str_root.replace("\\", '/'))[2:-2])
                    path = self.opt.save_path + "/GP/Symbolic classification"
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    os.makedirs(path)

                    string1,string2 = Symbolicclassification(csvname, path)

                    self.textBrowser.append(string1)
                    self.textBrowser.append(string2)

                    if self.opt.if_open == True:
                        str1 = (path + '/test_confusion.png').replace("/", "\\")
                        os.startfile(str1)
                        str2 = (path + '/train_confusion.png').replace("/", "\\")
                        os.startfile(str2)
                        str3 = (path + '/test_ROC.png').replace("/", "\\")
                        os.startfile(str3)
                        str4 = (path + '/train_ROC.png').replace("/", "\\")
                        os.startfile(str4)

                    QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)

                else:
                    QMessageBox.information(self, 'Hint', 'Not .csv file, please re-enter!',
                                            QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Please enter a file!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except Exception as e:
            print(e)








    # 预测集建立------------------------------------------------------------------------------------------------------
    # 预测集建立——导入预测集                          Prediction_Import virtual data (without label)
    def Prediction_Importvirtualdata(self):
        directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
        if len(directory_temp) > 0:
                str_root = str(directory_temp)
                f_csv = str_root.rfind('.csv')
                if f_csv != -1:                                                # 判断是不是.csv
                    self.opt.origin_path_2=str((str_root.replace("\\", '/'))[2:-2])
                    self.clear_state_Prediction()
                    self.import_prediction_dataset_state = True
                    QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                                QMessageBox.Close)
                else:
                    QMessageBox.information(self, 'Hint', 'Not .csv file, please re-enter!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
        else:
                QMessageBox.information(self, 'Hint', 'Please enter a file!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)

    # 预测集建立——预测集建立                          Prediction_Prediction construction (without label)


    # 预测集建立——选择模型                                          Prediction_Select machine learning model
    def import_model_dat(self):
        if self.import_prediction_dataset_state == True:
            directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
            if len(directory_temp) > 0:
                str_root = str(directory_temp)
                f_dat = str_root.rfind('.dat')
                if f_dat != -1:  # 判断是不是.dat
                    self.opt.origin_path_6 = str((str_root.replace("\\", '/'))[2:-2])

                    self.import_model_dat_state = True
                    QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                else:
                    QMessageBox.information(self, 'Hint', 'Not .dat file, please re-enter!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Please choose a model!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Do "Import virtual data (without label)"!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # 预测集建立——基于模型预测                         Prediction_Prediction generation (with label)
    def Prediction_Predictiongeneration(self):
        try:
            if self.import_prediction_dataset_state == True:
                if self. import_model_dat_state== True:
                    path = self.opt.save_path + "/Prediction/Prediction generation (with label)"
                    if os.path.exists(path) == False:
                        os.makedirs(path)

                    generate_file=dataML.model_modify_predict(self.opt.origin_path_2, path,self.opt.origin_path_6)


                    self.opt.prediction_visualization_path=generate_file
                    self.prediction_generation = True

                    QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                    if self.opt.if_open == True:
                        os.startfile(generate_file)
                else:
                    QMessageBox.information(self, 'Hint', 'Do "Select machine learning model"!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Do "Import virtual data (without label)"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except Exception as e:
            print(e)

    # 预测集建立——预测集可视化                         Prediction_Prediction visualization
    def Prediction_Prediction_visualization(self):
        try:
            if self. import_prediction_dataset_state== True:
                if self. import_model_dat_state== True:
                    if self.prediction_generation == True:
                        path = self.opt.save_path + "/Prediction/Prediction visualization"
                        csvname = self.opt.prediction_visualization_path
                        if os.path.exists(path):
                            shutil.rmtree(path)
                        os.makedirs(path)

                        value, ok = QtWidgets.QInputDialog.getText(self, "Prediction", "type column name for conversion:",
                                                                   QtWidgets.QLineEdit.Normal, "")
                        result=dataML.Visualization_for_classification(csvname, path, value)
                        if result==False:
                            QMessageBox.information(self, 'Hint', 'Column name not found!', QMessageBox.Ok | QMessageBox.Close,
                                                    QMessageBox.Close)
                        else:
                            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                                    QMessageBox.Close)
                            if self.opt.if_open == True:
                                os.startfile(path)
                    else:
                        QMessageBox.information(self, 'Hint', 'Do "Prediction generation (with label)"!', QMessageBox.Ok | QMessageBox.Close,
                                                QMessageBox.Close)
                else:
                    QMessageBox.information(self, 'Hint', 'Do "Select machine learning model"!', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Do "Import virtual data (without label)"!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except Exception as e:
            print(e)


    # shapley----------------------------------------------------------------------------------------------------
    # shapley_Model import
    def shapley_Modelimport(self):
        directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
        if len(directory_temp) > 0:
            str_root = str(directory_temp)
            f_dat = str_root.rfind('.dat')
            if f_dat != -1:  # 判断是不是.dat
                self.opt.origin_path_7 = str((str_root.replace("\\", '/'))[2:-2])

                #self.import_model_dat_state = True
                QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Not .dat file, please re-enter!',
                                        QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Please choose a model!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # shapley_Data import
    def shapley_Dataimport(self):
        directory_temp, filetype = QFileDialog.getOpenFileNames(self, "Select file")
        if len(directory_temp) > 0:
            str_root = str(directory_temp)
            f_csv = str_root.rfind('.csv')
            if f_csv != -1:  # 判断是不是.csv
                self.opt.origin_path_2 = str((str_root.replace("\\", '/'))[2:-2])
                #self.clear_state_Prediction()
                #self.import_prediction_dataset_state = True
                QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
            else:
                QMessageBox.information(self, 'Hint', 'Not .csv file, please re-enter!',
                                        QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        else:
            QMessageBox.information(self, 'Hint', 'Please enter a file!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    # shapley_Result
    def shapley_Result(self):
        try:
            path = self.opt.save_path + "/Shapley/Result"
            if os.path.exists(path) == False:
                os.makedirs(path)

            dataML.Result(self.opt.origin_path_2, path, self.opt.origin_path_7)

            if self.opt.if_open == True:
                str1 = (path + '/summary_plot.png').replace("/", "\\")
                os.startfile(str1)
                str2 = (path + '/Forceplot.png').replace("/", "\\")
                os.startfile(str2)
                str3 = (path + '/Feature_ranking_bar.png').replace("/", "\\")
                os.startfile(str3)
                str4 = (path + '/Waterfall.png').replace("/", "\\")
                os.startfile(str4)
                str5 = (path + '/decision_tree.png').replace("/", "\\")
                os.startfile(str5)

            QMessageBox.information(self, 'Hint', 'Completed!', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
        except Exception as e:
            print(e)

    """
    # 9                                                               # 1 9  and 1 3 9.3
    def nine_one(self):
        if self.training_test_set_path_state==True:
            self.nine_state=True
            path = self.opt.save_path + "/nine/one"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            value, ok = QtWidgets.QInputDialog.getDouble(self, "提示", "输入皮尔森阈值:", 0.55, -10000, 10000, 2)


            str1,data=dataML.gp_default(value)
            self.textBrowser.append(str1)
            self.textBrowser.append('\n***************************')
            list1 = list(data.columns)                                            # 把纵坐标名称竖着列
            str2 = ""
            for i in range(len(list1)):
                str_temp = "{str:^{len}}".format(str=str(list1[i]), len=24)
                str2 = str2 + "{str:^{len}}".format(str=str(i), len=24)+str_temp+"\n"
            self.textBrowser.append(str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, '提示', '完成', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
        else:
            QMessageBox.information(self, '提示', '第一步未运行', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    def nine_two(self):
        if self.training_test_set_path_state==True:
            self.nine_state=True
            path = self.opt.save_path + "/nine/two"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            value, ok = QtWidgets.QInputDialog.getDouble(self, "提示", "输入皮尔森阈值:", 0.55, -10000, 10000, 2)

            str1,data=dataML.gp_tan(value)
            self.textBrowser.append(str1)
            self.textBrowser.append('\n***************************')
            list1 = list(data.columns)  # 把纵坐标名称竖着列
            str2 = ""
            for i in range(len(list1)):
                str_temp = "{str:^{len}}".format(str=str(list1[i]), len=24)
                str2 = str2 + "{str:^{len}}".format(str=str(i), len=24) + str_temp + "\n"
            self.textBrowser.append(str2)
            self.textBrowser.append("*" * 150)

            QMessageBox.information(self, '提示', '完成', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
        else:
            QMessageBox.information(self, '提示', '第一步未运行', QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

    def nine_three(self):
        try:
            if self.training_test_set_path_state==True:
                if self.rfe_feature_selection_state == True:
                    self.nine_state=True
                    path = self.opt.save_path + "/nine/three"
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    os.makedirs(path)
                    value, ok = QtWidgets.QInputDialog.getDouble(self, "提示", "输入皮尔森阈值:", 0.6, -10000, 10000, 2)
                    dataML.tSR_default(value,path)
                    with open(path + "/data.txt", "r") as f:  # 从txt写入textBrowser
                        while True:
                            line = f.readline()  # 包括换行符
                            line = line[:-1]  # 去掉换行符
                            if line:
                                self.textBrowser.append(str(line))
                            else:
                                break
                    self.textBrowser.append("*" * 150)
                    QMessageBox.information(self, '提示', '完成', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
                else:
                    QMessageBox.information(self, '提示', '第三步未运行', QMessageBox.Ok | QMessageBox.Close,
                                            QMessageBox.Close)
            else:
                QMessageBox.information(self, '提示', '第一步未运行', QMessageBox.Ok | QMessageBox.Close,
                                        QMessageBox.Close)
        except Exception as e:
            print(e)
    """






if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = mywindow()

    ui.show()
    sys.exit(app.exec_())