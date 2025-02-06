import pandas as pd
import numpy as np
import os

folder_path = "Training_Anomalies_Station Data"
output_file_path = "train.csv"

merged_data = pd.DataFrame()

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        temp_data = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, temp_data], ignore_index=True)

merged_data['t'] = pd.to_datetime(merged_data['t'])
merged_data = merged_data.drop(columns=['latitude', 'longitude'])
filtered_data = merged_data[merged_data['t'] >= "1997-01-01"]

df_wide = filtered_data.pivot(index='t', columns='location', values='anomaly')
df_wide.reset_index(inplace=True)
df_wide.columns.name = None

df_wide.to_csv(output_file_path, index=False)