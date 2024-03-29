#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:04:06 2024

@author: ozanbaris
This script conducts the explanatory_data_analysis for cooling time impact of sensor count
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# File names
file_names = [ 'Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc']


# Initialize an empty DataFrame
df = pd.DataFrame()

# Load each file and append it to df
for file_name in file_names:
    # Load the .nc file
    data = xr.open_dataset(file_name)

    # Convert it to a DataFrame
    temp_df = data.to_dataframe()

    # Reset the index
    temp_df = temp_df.reset_index()

    # Append the data to df
    df = pd.concat([df, temp_df], ignore_index=True)

# Now df is a pandas DataFrame and you can perform any operation you want
print(df.head())

#%%
house_data = {}

unique_house_ids = df['id'].unique()

for house_id in unique_house_ids:
    house_data[house_id] = df[df['id'] == house_id]
    
#%%
    
metadata_df = pd.read_csv('meta_data.csv')
# Initialize an empty dictionary to store metadata for each house
metadata_houses = {}

# Iterate over each house_id in house_data
for house_id in unique_house_ids:
    # Find the row in metadata_df that matches the house_id
    metadata_row = metadata_df[metadata_df['Identifier'] == house_id]
    
    # If the house_id is found in metadata_df
    if not metadata_row.empty:
        # Extract the needed columns and convert to a dictionary
        # Here, we use .iloc[0] to get the first (and should be only) row as a Series, then to_dict()
        house_metadata = metadata_row.iloc[0].to_dict()
        
        # Assign this metadata dictionary to the corresponding house_id in metadata_houses
        metadata_houses[house_id] = house_metadata
    else:
        # Optional: Handle cases where no metadata is found for a house_id
        print(f"No metadata found for house_id {house_id}")

# Now metadata_houses contains the metadata for each house_id found in both datasets
#%%

# Sum the heating runtimes
df['TotalHeatingRuntime'] = df[['HeatingEquipmentStage1_RunTime', 'HeatingEquipmentStage2_RunTime', 'HeatingEquipmentStage3_RunTime']].sum(axis=1)

# Sum the cooling runtimes
df['TotalCoolingRuntime'] = df[['CoolingEquipmentStage1_RunTime', 'CoolingEquipmentStage2_RunTime']].sum(axis=1)


total_runtimes_by_house = df.groupby('id')[['TotalHeatingRuntime', 'TotalCoolingRuntime']].sum()

# Now, total_runtimes_by_house is a DataFrame with each house_id and its total heating and cooling runtimes
print(total_runtimes_by_house.head())
#%%
# Iterate over the total_runtimes_by_house DataFrame
for house_id, row in total_runtimes_by_house.iterrows():
    # Update metadata_houses with TotalHeatingRuntime and TotalCoolingRuntime
    if house_id in metadata_houses:  # Ensure the house_id exists in metadata_houses
        metadata_houses[house_id]['TotalHeatingRuntime'] = row['TotalHeatingRuntime']
        metadata_houses[house_id]['TotalCoolingRuntime'] = row['TotalCoolingRuntime']
# Compute the average outdoor temperature for each house_id
average_outdoor_temp_by_house = df.groupby('id')['Outdoor_Temperature'].mean()

for house_id, avg_temp in average_outdoor_temp_by_house.items():
    if house_id in metadata_houses:  # Ensure the house_id exists in metadata_houses
        metadata_houses[house_id]['AverageOutdoorTemperature'] = avg_temp



metadata_houses_df = pd.DataFrame.from_dict(metadata_houses, orient='index')

# Ensure the DataFrame looks correct
print(metadata_houses_df.head())

# Step 2: Group by 'ProvinceState'
grouped_by_provincestate = metadata_houses_df.groupby('ProvinceState')

# Example: To see the size of each group (i.e., count of houses in each province/state)
print(grouped_by_provincestate.size())

mean_heating_runtime_by_province = grouped_by_provincestate['TotalHeatingRuntime'].mean()
print(mean_heating_runtime_by_province)
#%%
# Define remote sensor columns
remote_sensor_columns = [
    'RemoteSensor1_Temperature',
    'RemoteSensor2_Temperature',
    'RemoteSensor3_Temperature',
    'RemoteSensor4_Temperature',
    'RemoteSensor5_Temperature'
]

# Initialize a dictionary to hold the count of sensors for each house
sensors_count = {}

for house_id, data in house_data.items():
    # Count how many of these columns have mostly non-NaN values for each house
    # We'll consider "mostly" as more than 70% non-NaN values in the column
    non_nan_count = sum(data[column].notna().mean() > 0.7 for column in remote_sensor_columns)
    sensors_count[house_id] = non_nan_count

    # Update the 'Number of Remote Sensors' in metadata_houses_df for the corresponding house_id
    if house_id in metadata_houses_df.index:
        metadata_houses_df.loc[house_id, 'Number of Remote Sensors'] = non_nan_count

# Now, print the number of houses with a certain number of sensors
sensor_distribution = metadata_houses_df['Number of Remote Sensors'].value_counts().sort_index()
print(sensor_distribution)
#%%

# Ensure 'Number of Remote Sensors' is treated as a categorical variable
metadata_houses_df['Number of Remote Sensors'] = pd.Categorical(metadata_houses_df['Number of Remote Sensors'], categories=[0, 1, 2, 3, 4, 5])

# Plotting
plt.figure(figsize=(20, 10))

# Use seaborn to create the boxplot
sns.boxplot(x='ProvinceState', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')

#plt.title('Distribution of Total Cooling Time by Number of Sensors and State')
plt.xticks(rotation=0)  # Set x-tick labels vertically
plt.xlabel('State')
plt.ylabel('Total Cooling Time')
plt.legend(title='Number of Sensors')

plt.show()
#%%

# Define the bins and labels
bins = [68, 72, 76, 80, 84]
bins.insert(0, -np.inf)  # To include everything below the first bin
bins.append(np.inf)  # To include everything above the last bin
labels = ['< 68', '68-72', '72-76', '76-80', '80-84', '84+']

# Categorize 'AverageOutdoorTemperature' into 'TemperatureBins'
metadata_houses_df['TemperatureBins'] = pd.cut(metadata_houses_df['AverageOutdoorTemperature'], bins=bins, labels=labels)

# Ensure 'Number of Remote Sensors' is treated as a categorical variable
metadata_houses_df['Number of Remote Sensors'] = pd.Categorical(metadata_houses_df['Number of Remote Sensors'], categories=[0, 1, 2, 3, 4, 5])

# Plotting
plt.figure(figsize=(20, 10))

# Use seaborn to create the boxplot
sns.boxplot(x='TemperatureBins', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')

#plt.title('Distribution of Total Cooling Time by Number of Sensors and Average Outdoor Temperature Bins')
plt.xticks(rotation=0)  # Ensure x-tick labels are vertical for readability
plt.xlabel('Average Outdoor Temperature Bins')
plt.ylabel('Total Cooling Time')
plt.legend(title='Number of Sensors', loc='upper right')

plt.show()

#%%
# First, remove houses with floor area smaller than 100 and count them
initial_count = len(metadata_houses_df)
metadata_houses_df = metadata_houses_df[metadata_houses_df['Floor Area [ft2]'] >= 100]
removed_count = initial_count - len(metadata_houses_df)

# Define the bins and labels for floor area
bins = [100, 1000, 2000, 3000, 4000, float('inf')]
labels = ['100-1000', '1000-2000', '2000-3000', '3000-4000', '4000+']

# Categorize 'Floor Area [ft2]' into 'FloorAreaBins'
metadata_houses_df['FloorAreaBins'] = pd.cut(metadata_houses_df['Floor Area [ft2]'], bins=bins, labels=labels)

# Plotting
plt.figure(figsize=(20, 10))

# Use seaborn to create the boxplot
sns.boxplot(x='FloorAreaBins', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')

#plt.title('Distribution of Total Cooling Time by Number of Sensors and Floor Area Bins')
plt.xticks(rotation=0)  # Ensure x-tick labels are vertical for readability
plt.xlabel('Floor Area')
plt.ylabel('Total Cooling Time')
plt.legend(title='Number of Sensors', loc='upper right')

plt.show()

# Print the number of houses removed
removed_count
#%%

# Convert 'Number of Occupants' to a string to represent categories, including "5+" for 5 or more
metadata_houses_df['OccupantsCategory'] = np.select(
    [metadata_houses_df['Number of Occupants'] < 5, metadata_houses_df['Number of Occupants'] >= 5],
    [metadata_houses_df['Number of Occupants'], '5+'],
    default='Unknown'
)


plt.figure(figsize=(12, 8))

# Create the boxplot
sns.boxplot(x='OccupantsCategory', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')

plt.xticks(rotation=0)  # Set x-tick labels vertically for better readability
plt.xlabel('Number of Occupants')
plt.ylabel('Total Cooling Time')
plt.legend(title='Number of Sensors', loc='upper right')

plt.show()

#%%

fig, axes = plt.subplots(4, 1, figsize=(12, 20))

# Attempt direct plt.subplots_adjust with tighter spacing
plt.subplots_adjust(hspace=0.2)  # Try adjusting this value as needed

#  By State
sns.boxplot(ax=axes[0], x='ProvinceState', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')
axes[0].set_xlabel('State')
axes[0].set_ylabel('Total Cooling Time')
# Place the legend outside the plot on top with reduced title font size
legend = axes[0].legend(title='Number of Sensors', loc='upper center', bbox_to_anchor=(0.5, 1.58), ncol=6)
legend.set_title('Number of Sensors', prop={'size': 18})  # Adjust title font size

#  By Average Outdoor Temperature Bins
sns.boxplot(ax=axes[1], x='TemperatureBins', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')
axes[1].set_xlabel('Average Outdoor Temperature')
axes[1].set_ylabel('Total Cooling Time')
axes[1].get_legend().remove()

#  By Floor Area Bins
sns.boxplot(ax=axes[2], x='FloorAreaBins', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')
axes[2].set_xlabel('Floor Area')
axes[2].set_ylabel('Total Cooling Time')
axes[2].get_legend().remove()

#  By Number of Occupants, sorted
metadata_houses_df['OccupantsCategory'] = pd.Categorical(metadata_houses_df['OccupantsCategory'], 
                                                         categories=['1', '2', '3', '4', '5+'], 
                                                         ordered=True)
sns.boxplot(ax=axes[3], x='OccupantsCategory', y='TotalCoolingRuntime', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')
axes[3].set_xlabel('Number of Occupants')
axes[3].set_ylabel('Total Cooling Time')
axes[3].get_legend().remove()

# If using tight_layout, adjust its parameters to see if it helps
plt.tight_layout(pad=1.0, h_pad=0.01, w_pad=0.2, rect=[0, 0, 1, 0.90])
plt.savefig('energy_sensors.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%
# Calculate cooling time per unit ft^2
metadata_houses_df['CoolingTimePerUnitFt2'] = metadata_houses_df['TotalCoolingRuntime'] / metadata_houses_df['Floor Area [ft2]']

# Create a figure with two subplots (vertically arranged)
fig, axes = plt.subplots(2, 1, figsize=(15, 20))

# First Plot: By Temperature Bins
sns.boxplot(ax=axes[0], x='TemperatureBins', y='CoolingTimePerUnitFt2', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')
axes[0].set_xlabel('Average Outdoor Temperature')
axes[0].set_ylabel('Cooling Time per Unit Ft²')
axes[0].tick_params(axis='x', rotation=0)  # Rotate x-ticks for better readability
axes[0].legend(title='Number of Remote Sensors', ncol=2)  # Remove the legend from the first plot to avoid repetition

# Second Plot: By ProvinceState
sns.boxplot(ax=axes[1], x='ProvinceState', y='CoolingTimePerUnitFt2', hue='Number of Remote Sensors',
            data=metadata_houses_df, showfliers=False, palette='Spectral')
axes[1].set_xlabel('State')
axes[1].set_ylabel('Cooling Time per Unit Ft²')
axes[1].tick_params(axis='x', rotation=0)  # Rotate x-ticks for better readability
axes[1].get_legend().remove()
# Place a single legend outside the top plot
handles, labels = axes[1].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=6, title='Number of Remote Sensors', bbox_to_anchor=(0.5, 1.05))
plt.savefig('cooling_per_unit.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

