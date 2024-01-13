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
        # df_combined = pd.concat(selected_columns.values(), axis=1)
        # df_combined.to_csv('selected_columns.csv', index=False)
        selected_columns[suffix].to_csv(f'{suffix}_selected_columns.csv', index=False)

    # 获取未被选中的列
    unselected_columns = df.drop(columns=[col for cols in selected_columns.values() for col in cols.columns])

    # 保存未被选中的列到 CSV 文件
    unselected_columns.to_csv('unselected_columns.csv', index=False)


    return selected_columns

# 用法示例
file_path = 'Formula-Smiles-lsy-reg.csv'
suffixes = ['InorganicFormula', 'OrganicSmiles']
selected_columns = extract_and_store_columns(file_path, suffixes)


original_data = pd.DataFrame(pd.read_csv('InorganicFormula_selected_columns.csv'))

original_data = pd.DataFrame(pd.read_csv('InorganicFormula_selected_columns.csv'))
import pandas as pd

# 假设 original_data 是您的原始数据集
# 创建一个空字典，用于存储新的数据集
new_datasets = {}

# 遍历原始数据集的每一列
for col_name in original_data.columns:
    # 创建新的数据集，将当前列命名为 'Name'
    new_dataset = pd.DataFrame({'Name': original_data[col_name]})

    # 将新数据集存储在字典中，字典的键是 'data1'，'data2'，依此类推
    new_datasets['data' + str(len(new_datasets) + 1)] = new_dataset

# 打印或使用新的数据集
for key, value in new_datasets.items():
    print(f"{key}:\n{value}\n")

import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.orbital import AtomicOrbitals
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.composition.element import ElementFraction

# 假设 new_datasets 是包含拆分数据集的字典，如 'data1', 'data2', ...
# 每个数据集中应该有 'Name' 列

# 初始化 StrToComposition
str_to_comp = StrToComposition(target_col_id='composition')

# 初始化 AtomicOrbitals
comp_to_orbital = AtomicOrbitals()

# 初始化 ElementProperty
features_element_property = ['Number', 'MendeleevNumber', 'AtomicWeight', 'MeltingT',
                             'Column', 'Row', 'CovalentRadius', 'Electronegativity',
                             'NsValence', 'NpValence', 'NdValence', 'NfValence', 'NValence',
                             'NsUnfilled', 'NpUnfilled', 'NdUnfilled', 'NfUnfilled', 'NUnfilled',
                             'GSvolume_pa', 'GSbandgap', 'GSmagmom', 'SpaceGroupNumber']
stats_element_property = ['mean', 'minimum', 'maximum', 'range', 'avg_dev', 'mode']
element_property_featurizer = ElementProperty(data_source='magpie', features=features_element_property,
                                              stats=stats_element_property)

# 初始化 ElementFraction
element_fraction = ElementFraction()

# 用于存储特征转换后的数据集
result_datasets = {}

# 遍历拆分的数据集
for i, (key, dataset) in enumerate(new_datasets.items(), start=1):
    # 特征转换1: StrToComposition
    df_comp = str_to_comp.featurize_dataframe(dataset, col_id='Name')

    # 特征转换2: AtomicOrbitals
    orbital_features = comp_to_orbital.featurize_dataframe(df_comp, col_id='composition')
    orbital_features = orbital_features.iloc[:, [4, 7, 8]]  # 选择感兴趣的列

    # 特征转换3: ElementProperty
    element_property_features = element_property_featurizer.featurize_dataframe(df_comp, col_id='composition')
    element_property_features = element_property_features.iloc[:, 2:-1]  # 选择感兴趣的列

    # 特征转换4: ElementFraction
    element_fraction_features = element_fraction.featurize_dataframe(df_comp, col_id='composition')
    element_fraction_features = element_fraction_features.iloc[:, 2:-1]  # 选择感兴趣的列

    # 添加前缀
    prefix_orbital = f'inorganic_formula_{i}_orbital_'
    orbital_features = orbital_features.add_prefix(prefix_orbital)

    prefix_element_property = f'inorganic_formula_{i}_element_property_'
    element_property_features = element_property_features.add_prefix(prefix_element_property)

    prefix_element_fraction = f'inorganic_formula_{i}_element_fraction_'
    element_fraction_features = element_fraction_features.add_prefix(prefix_element_fraction)

    # 合并特征转换后的数据集
    result_datasets[key] = pd.concat([orbital_features, element_property_features, element_fraction_features], axis=1)

# 合并所有数据集
merged_result = pd.concat(result_datasets.values(), axis=1)

# 将合并后的结果保存为 CSV 文件
merged_result.to_csv('merged_result.csv', index=False)

# 打印或使用合并后的结果
print(merged_result)

# 有机部分
original_data2 = pd.DataFrame(pd.read_csv('OrganicSmiles_selected_columns.csv'))

new_datasets2 = {}

# 遍历原始数据集的每一列
for col_name in original_data2.columns:
    # 创建新的数据集，将当前列命名为 'Name'
    new_dataset2 = pd.DataFrame({'Name': original_data2[col_name]})

    # 将新数据集存储在字典中，字典的键是 'organic_data1'，'organic_data2'，依此类推
    new_datasets2['organic_data' + str(len(new_datasets2) + 1)] = new_dataset2

# 打印或使用新的数据集
for key, value in new_datasets2.items():
    print(f"{key}:\n{value}\n")

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import AllChem

# 假设 new_datasets2 是包含拆分数据集的字典，如 'organic_data1', 'organic_data2', ...
# 每个数据集中应该有 'Name' 列

# 用于存储 RDKit 特征化后的数据集
rdkit_datasets = {}

# 遍历拆分的数据集
for key, dataset in new_datasets2.items():
    # 特征化 RDKit
    rdkit_features = dataset['Name'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2))

    # 将 RDKit 指纹转换为 DataFrame
    rdkit_features_df = pd.DataFrame(list(rdkit_features.apply(lambda x: np.frombuffer(x.ToBinary(), dtype=np.uint8))))

    # 添加前缀
    prefix_rdkit = f'{key}_rdkit_'
    rdkit_features_df = rdkit_features_df.add_prefix(prefix_rdkit)

    # 存储特征化后的数据集
    rdkit_datasets[key] = rdkit_features_df

# 合并 RDKit 特征化后的数据集
merged_rdkit_result = pd.concat(rdkit_datasets.values(), axis=1)
merged_rdkit_result.fillna(0, inplace=True)
# 将合并后的结果保存为 CSV 文件
merged_rdkit_result.to_csv('merged_rdkit_result.csv', index=False)

# 打印或使用合并后的 RDKit 特征化结果
print(merged_rdkit_result)


unselected_columns=pd.DataFrame(pd.read_csv('unselected_columns.csv'))

# 合并前重置索引
merged_result.reset_index(drop=True, inplace=True)
merged_rdkit_result.reset_index(drop=True, inplace=True)
unselected_columns.reset_index(drop=True, inplace=True)

# 合并三个数据集
all_merged_data = pd.concat([merged_result, merged_rdkit_result, unselected_columns], axis=1)

# 将合并后的结果保存为 CSV 文件
all_merged_data.to_csv('test_train_dataset.csv', index=False)
