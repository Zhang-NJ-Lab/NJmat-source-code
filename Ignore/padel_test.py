import pandas as pd
def smiles_csv_pydel(name2):
    global data2
    data2 = pd.read_csv(name2)
    print(data2.iloc[:,0])
    return data2

# 0.1.1.2 pydel描述符生成
def pydel_featurizer():
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
    data2b.to_csv("pydel_featurizer_output.csv")
    return data2b


smiles_csv_pydel('Organic_rdkit.csv')
pydel_featurizer()