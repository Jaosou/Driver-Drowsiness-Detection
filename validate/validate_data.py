import pandas as pd
import numpy as np


file_path = 'C:/Project/End/Code/Data/round/ear_data_round2.csv'
data = pd.read_csv(file_path)

range_ratio = 0.45

ear_value_left = (pd.to_numeric(data['ear_value_left'], errors='coerce')).dropna()
ear_value_right = pd.to_numeric(data['ear_value_right'], errors='coerce').dropna()

data = data[data['ear_value_left'] < 0.45]
data = data[data['ear_value_right'] < 0.45]

data.to_csv(file_path, index=False)



