import numpy as np
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(),r'acc_maps.csv'))

print(df.shape)

for i in range(len(df)):
    if (i % 2 !=0):
        df.drop(index=i,inplace=True)

print(df.shape)


df.to_csv('output.csv',index=False)