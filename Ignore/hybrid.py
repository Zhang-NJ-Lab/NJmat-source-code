# import pandas as pd
#
# # 创建一个示例DataFrame
# data = {'apple': [1, 2, 3], 'banana': [4, 5, 6], 'grape': [7, 8, 9]}
# df = pd.DataFrame(data)
#
# # 假设你知道列名以 'ple' 结尾，你可以这样选择列
# columns_end_with = 'ple'
#
# filtered_columns = df.filter(regex=f'{columns_end_with}$')
# print(filtered_columns)
#
import pandas as pd

# def select_columns_by_suffix(df, suffix):
#     filtered_columns = df.filter(regex=f'{suffix}$')
#     return filtered_columns
#
# def read_csv_and_select_columns(file_path, suffix):
#     # 读取 CSV 文件
#     df = pd.read_csv(file_path)
#
#     # 调用列选择函数
#     selected_columns = select_columns_by_suffix(df, suffix)
#     return selected_columns
#
# # 文件路径和要匹配的后缀
# file_path = 'Inorganic-Organic-Hybrid-template.csv'
# suffix = 'InorganicFormula'
# #
# # 调用函数并传入文件路径和后缀
# result = read_csv_and_select_columns(file_path, suffix)
# print('The following columns correpond to materials formulas (inorganic variables)')
# print('********************************************************************************')
# print(result)
#
import pandas as pd

def select_columns_by_suffix(df, suffix):
    filtered_columns = df.filter(regex=f'{suffix}$')
    return filtered_columns

def extract_and_store_columns(csv_file, suffixes):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    selected_columns = {}
    for suffix in suffixes:
        selected_columns[suffix] = select_columns_by_suffix(df, suffix)
        print('********************************************************************************')
        print(f"Columns ending with '{suffix}':")
        print('********************************************************************************')
        print(selected_columns[suffix])
        # 如果需要保存到新的DataFrame中，取消注释下一行
        global df_selected
        df_selected = pd.concat(selected_columns, axis=1)

    return selected_columns

# 用法示例
file_path = 'Inorganic-Organic-Hybrid-template.csv'
suffixes = ['InorganicFormula', 'OrganicSmiles']
selected_columns = extract_and_store_columns(file_path, suffixes)
# print(f"Columns ending with '{suffix}':")
# print('********************************************************************************')
# print(df_selected)
#
#
# # SMILES
# import pandas as pd
#
# def select_columns_by_suffix(df, suffix):
#     filtered_columns = df.filter(regex=f'{suffix}$')
#     return filtered_columns
#
# def read_csv_and_select_columns(file_path, suffix):
#     # 读取 CSV 文件
#     df = pd.read_csv(file_path)
#
#     # 调用列选择函数
#     selected_columns = select_columns_by_suffix(df, suffix)
#     return selected_columns
#
# # 文件路径和要匹配的后缀
# file_path = 'Inorganic-Organic-Hybrid-template.csv'
# suffix = 'InorganicFormula'
# #
# # 调用函数并传入文件路径和后缀
# result = read_csv_and_select_columns(file_path, suffix)
# print('The following columns correpond to materials formulas (inorganic variables)')
# print('********************************************************************************')
# print(result)
# #
# #
# #

'''
def Hybrid_organic_inorganic_csv(name34):                        # one-hot matiminer import
    import pandas as pd
    global data34
    data4 = pd.read_csv(name34)
    print(data34)
    return data34

Hybrid_organic_inorganic_csv

# 0.2.2 matminer无机材料（类独热编码）描述符生成，102维
# 例如(Fe2AgCu2)O3, Fe2O3, Cs3PbI3, MoS2, CuInGaSe, Si, TiO2等
def Hybrid_organic_inorganic_featurizer(path):                     # one-hot matiminer featurization
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
    element_fraction_labels = ef.feature_labels()
    print(element_fraction_labels)
    data8.columns = element_fraction_labels
    print(data8)
    # 特征存入pydel_featurizer.csv
    data8.to_csv(path+"/inorganic_featurizer_output.csv",index=None)
    return data8,element_fraction_labels
'''