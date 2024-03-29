#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:19:00 2024

@author: ozanbaris
This script preprocess the ecobee data to save it in dta format for further analysis on R.
"""
import xarray as xr
import pandas as pd

# File names
file_names = ['Jan_clean.nc', 'Feb_clean.nc', 'Mar_clean.nc', 'Apr_clean.nc', 'May_clean.nc', 
              'Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc', 'Sep_clean.nc', 'Oct_clean.nc', 
              'Nov_clean.nc', 'Dec_clean.nc']

# Initialize an empty DataFrame to store the final dataset
final_df = pd.DataFrame()

# Remote sensor temperature columns
temp_columns = [
    'RemoteSensor1_Temperature',
    'RemoteSensor2_Temperature',
    'RemoteSensor3_Temperature',
    'RemoteSensor4_Temperature',
    'RemoteSensor5_Temperature',
    'Outdoor_Temperature'  # Including Outdoor_Temperature
]

def process_and_resample(file_name):
    data = xr.open_dataset(file_name)
    df = data.to_dataframe().reset_index()
    
    # Perform necessary column calculations
    df['CoolingRuntime'] = df[['CoolingEquipmentStage1_RunTime', 'CoolingEquipmentStage2_RunTime']].sum(axis=1)
    df['HeatingRuntime'] = df[['HeatingEquipmentStage1_RunTime', 'HeatingEquipmentStage2_RunTime', 'HeatingEquipmentStage3_RunTime']].sum(axis=1)
    
    # Drop the original CoolingEquipmentStage and HeatingEquipmentStage columns
    columns_to_drop = [col for col in df.columns if "CoolingEquipmentStage" in col or "HeatingEquipmentStage" in col]
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Ensure 'time' is a datetime column
    df['time'] = pd.to_datetime(df['time'])
    
    # Adjust the aggregation dictionary to take the mean for temperature columns and sum for 'Runtime' columns
    agg_dict = {col: 'mean' if col in temp_columns else 'sum' for col in df.columns if 'Runtime' in col}
    agg_dict.update({col: 'mean' for col in temp_columns if col not in agg_dict})  # Add temperature columns if not already included
    
    # Group by 'id', then resample and aggregate
    grouped = df.groupby('id')
    resampled_list = []
    for name, group in grouped:
        group.set_index('time', inplace=True)
        resampled = group.resample('H').agg(agg_dict)
        resampled['id'] = name  # Add 'id' back to the dataframe
        resampled_list.append(resampled)
        
    # Concatenate all resampled dataframes
    return pd.concat(resampled_list).reset_index()

# Process each file and concatenate the results
for file_name in file_names:
    resampled_df = process_and_resample(file_name)
    final_df = pd.concat([final_df, resampled_df], ignore_index=True)

print(final_df.head())



#%%
# List of temperature columns to check for non-NaN values
temp_columns = [
    'RemoteSensor1_Temperature',
    'RemoteSensor2_Temperature',
    'RemoteSensor3_Temperature',
    'RemoteSensor4_Temperature',
    'RemoteSensor5_Temperature'
]

# Compute num_sensors by counting non-NaN values across the specified temperature columns for each row
final_df['num_sensors'] = final_df[temp_columns].notna().sum(axis=1)
#%%
# Save the modified DataFrame to a CSV file
final_df.to_csv('final_df_reduced.csv', index=False)

print("DataFrame saved with 'num_sensors' column computed.")

#%% After the first run, hourly data can be read instead of running the cells above.

file_path = 'final_df_reduced.csv'

# Read the dataset into a DataFrame
final_df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify it's read correctly
final_df.head()



#%%

sensor_dummies = pd.get_dummies(final_df['num_sensors'], prefix='sensor')
final_df = pd.concat([final_df, sensor_dummies], axis=1)

# Convert 'time' column to datetime format (existing step)
final_df['time'] = pd.to_datetime(final_df['time'])

# Convert 'Outdoor_Temperature' to numeric, filling missing values with forward fill (existing step)
final_df['Outdoor_Temperature'] = pd.to_numeric(final_df['Outdoor_Temperature'], errors='coerce').fillna(method='ffill')

# Calculate cooling duty cycle from 'CoolingRuntime' (existing step)
final_df['cooling_duty_cycle'] = final_df['CoolingRuntime'] / 3600  # Convert seconds to hours
final_df['cooling_duty_cycle'] = final_df['cooling_duty_cycle'].fillna(0)



#%%
temp_columns = [
    'RemoteSensor1_Temperature',
    'RemoteSensor2_Temperature',
    'RemoteSensor3_Temperature',
    'RemoteSensor4_Temperature',
    'RemoteSensor5_Temperature'
]

final_df.drop(columns=temp_columns, inplace=True)

#%%
final_df = final_df[final_df['Outdoor_Temperature'] > 60]

# Creating interaction terms between sensor dummies and outdoor temperature
for column in sensor_dummies.columns:
    final_df[f'{column}_temp_interaction'] = final_df[column] * final_df['Outdoor_Temperature']
    
#%%

# Renaming interaction columns to a shorter format
for i in range(6):  
    old_name = f'sensor_{i}_temp_interaction'
    new_name = f's{i}_T'
    final_df.rename(columns={old_name: new_name}, inplace=True)

# Print the updated DataFrame columns to verify the changes
print(final_df.columns)

# Save the DataFrame to a DTA file
final_df.to_stata('final_df_unbinned.dta')

print("DataFrame saved with abbreviated column names.")


