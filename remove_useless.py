import numpy as np
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(),r'output.csv'))

tmp = df.columns.to_list()
print(len(df.columns.to_list()))


#find all static data and remove

static_data = []

for item in tmp:
    if item.startswith('Static'):
        static_data.append(item)

df = df.drop(columns=static_data)


#remove graphics car coordinates since it is a vector contains x,y,z and not well formated.
test = df["Graphics_car_coordinates"]
df = df.drop(["Graphics_car_coordinates"],axis=1)

xs = []
ys = []
zs = []

# split the vec<x,y,z> in graphics_car_coordinates into x,y,z

for item in test:
    #tmp = test.iloc[0]
    vec = item[item.find('(')+1:item.find(')')]
    x = vec[vec.find('x')+2:vec.find(',')]
    y = vec[vec.find('y')+2:vec.find('z')-2]
    z = vec[vec.find('z')+2:]
    xs.append(x)
    ys.append(y)
    zs.append(z)
# add the data into the dataframce
df["Graphics_car_coordinate_x"] = xs
df["Graphics_car_coordinate_y"] = ys
df["Graphics_car_coordinate_z"] = zs

df.to_csv('output_clean_split.csv',index=False)


