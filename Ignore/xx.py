# import pandas as pd
#
#
# data = {"name": ["Jaide", "Aaron", "Adam"], "Age": [12, 34, 98]}
# dataFrame = pd.DataFrame(data)
# print(dataFrame)
#
# print("**********")
#
# print(dataFrame.columns[0])

import pandas as pd

# 创建一个示例DataFrame
data = {'apple': [1, 2, 3], 'banana': [4, 5, 6], 'grape': [7, 8, 9]}
df = pd.DataFrame(data)

# 假设你知道列名以 'ple' 结尾，你可以这样选择列
columns_end_with = 'ple'

filtered_columns = df.filter(regex=f'{columns_end_with}$')
print(filtered_columns)