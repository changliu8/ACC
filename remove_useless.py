import numpy as np
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(),r'output.csv'))

tmp = df.columns.to_list()
print(len(df.columns.to_list()))


static_data = []

for item in tmp:
    if item.startswith('Static'):
        static_data.append(item)

df = df.drop(columns=static_data)

print(len(df.columns.to_list()))

df.to_csv('output_clean.csv',index=False)
