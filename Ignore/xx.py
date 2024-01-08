def inorganic_featurizer(path):
    import pandas as pd
    from matminer.featurizers.composition.element import ElementFraction
    from pymatgen.core import Composition

    ef = ElementFraction()
    list4 = list(map(lambda x: Composition(x), data4.iloc[:, 0]))
    data7 = pd.DataFrame()
    for i in range(0, len(data4.index)):
        data7 = pd.concat([data7, pd.DataFrame(ef.featurize(list4[i])).T])
    data8 = data7.reset_index()
    data8 = data8.iloc[:, 1:]
    element_fraction_labels = ef.feature_labels()
    print(element_fraction_labels)
    data8.columns = element_fraction_labels
    print(data8)
    # 特征存入pydel_featurizer.csv
    data8.to_csv(path + "/inorganic_featurizer_output.csv", index=None)
    return data8, element_fraction_labels


def inorganic_csv(name4):
    import pandas as pd
    global data4
    data4 = pd.read_csv(name4)
    print(data4)
    return data4


path = "D:/new_generate/Descriptor generation/Inorganic molecular descriptors/matminer"
data4=inorganic_csv("D:/NJmatML_project_zl/datasets/Inorganic_formula.csv")

data8,element_fraction_labels=inorganic_featurizer(path)
print(1)
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


