import numpy as np
import pandas as pd

df = pd.read_csv("output_clean_split.csv")


Y = df["Physics_speed_kmh"]
X = df.drop(["Physics_speed_kmh"],axis=1)




