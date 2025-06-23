import numpy as np
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(),r'output.csv'))

test = df["Graphics_car_coordinates"]

tmp = test.iloc[0]
vec = tmp[tmp.find('(')+1:tmp.find(')')]
x = vec[vec.find('x')+2:vec.find(',')]
y = vec[vec.find('y')+2:vec.find('z')-2]
z = vec[vec.find('z')+2:]
print(x,y,z)