import pandas as pd


data = {"name": ["Jaide", "Aaron", "Adam"], "Age": [12, 34, 98]}
dataFrame = pd.DataFrame(data)
print(dataFrame)

print("**********")

print(dataFrame.columns[0])