import numpy as np
import pandas as pd
import os


df = pd.read_csv(os.path.join(os.getcwd(),r'output_clean.csv'))

test = df["Graphics_car_coordinates"]
print(len(df.columns.to_list()))
df = df.drop(["Graphics_car_coordinates"],axis=1)
print(len(df.columns.to_list()))


xs = []
ys = []
zs = []

for item in test:
    #tmp = test.iloc[0]
    vec = item[item.find('(')+1:item.find(')')]
    x = vec[vec.find('x')+2:vec.find(',')]
    y = vec[vec.find('y')+2:vec.find('z')-2]
    z = vec[vec.find('z')+2:]
    xs.append(x)
    ys.append(y)
    zs.append(z)

df["Graphics_car_coordinate_x"] = xs
df["Graphics_car_coordinate_y"] = ys
df["Graphics_car_coordinate_z"] = zs

print(len(df.columns.to_list()))

print(len(xs),len(ys),len(zs))

df.to_csv('output_clean_split.csv',index=False)
