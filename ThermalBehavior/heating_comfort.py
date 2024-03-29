#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:20:11 2023

@author: ozanbaris
"This is the script for the analysis regarding the efficiency of averaging on thermal comfort for cooling season."
"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# File names
file_names = ['Jan_clean.nc', 'Feb_clean.nc', 'Mar_clean.nc', 'Apr_clean.nc', 
              'May_clean.nc', 
              'Sep_clean.nc', 'Oct_clean.nc', 'Nov_clean.nc', 'Dec_clean.nc']

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
    df_heat = pd.concat([df, temp_df], ignore_index=True)

# Now df is a pandas DataFrame and you can perform any operation you want
print(df_heat.head())

#%%
house_heat_data = {}

unique_house_ids = df_heat['id'].unique()

for house_id in unique_house_ids:
    single_house_heat_data = df_heat[df_heat['id'] == house_id]
    
    # Subtract 1 year from 'time' for months September through December
    single_house_heat_data.loc[single_house_heat_data['time'].dt.month >= 9, 'time'] = single_house_heat_data.loc[single_house_heat_data['time'].dt.month >= 9, 'time'].apply(lambda x: x.replace(year=x.year-1))
    
    # Sort DataFrame by 'time'
    single_house_heat_data.sort_values(by='time', inplace=True)
    
    house_heat_data[house_id] = single_house_heat_data
#%%# Prepare dictionaries to store cooling and heating season data for each house
heating_season_dict = {}
no_heating_houses = 0

for house_id, single_orig_data in house_heat_data.items():
    # Identify when the HVAC system is in cooling or heating mode
    single_house_data = single_orig_data.copy()
    single_house_data.set_index('time', inplace=True)
    single_house_data['Cooling_Mode'] = single_house_data['CoolingEquipmentStage1_RunTime'].notna()
    single_house_data['Heating_Mode'] = single_house_data['HeatingEquipmentStage1_RunTime'].notna()

    # Create boolean Series with True when a switch from cooling to heating or from heating to cooling happens
    switches = single_house_data['Cooling_Mode'].astype(int).diff().abs() == 1

    # Identify the start and end times of each period
    periods = single_house_data[switches].index

    # Extract heating and cooling data
    heating_data = single_house_data[single_house_data['Heating_Mode']]
    cooling_data = single_house_data[single_house_data['Cooling_Mode']]

    # Ensure that each period starts with heating and ends with cooling
    if not cooling_data.empty and periods.size > 0 and periods[0] < cooling_data.index.min():
        periods = periods[1:]
    if not heating_data.empty and periods.size > 0 and periods[-1] > heating_data.index.max():
        periods = periods[:-1]

    # Create tuples of each period's start and end times
    periods_tuples = list(zip(periods[::2], periods[1::2])) if periods.size > 0 else []

    # If there are transitions, identify the longest period (i.e., summer)
    if periods_tuples:
        winter_period = max(periods_tuples, key=lambda x: x[1] - x[0])
    else:
        # If there are no transitions, use the first and last instances of cooling as the summer period
        if not heating_data.empty:  # If there is cooling data
            winter_period = (heating_data.index.min(), heating_data.index.max())
        else:
            # If there is no cooling data, skip this house and add to the counter of no cooling houses
            no_heating_houses += 1
            continue

    # Extract summer data
    winter_data = single_house_data.loc[winter_period[0]:winter_period[1]]

    # Store in the dictionary
    heating_season_dict[house_id] = winter_data

# Print the number of houses with no cooling
print(f"There are {no_heating_houses} houses without any heating.")



#%%
# Create empty dictionaries for each category
heat_one_houses = {}
heat_two_houses = {}
heat_three_houses = {}
heat_four_houses = {}
heat_five_houses = {}

for house_id, data in heating_season_dict.items():
    # Count the number of non-empty temperature sensors
    num_sensors = sum([1 for i in range(1, 6) if not np.isnan(data[f'RemoteSensor{i}_Temperature']).all()])
    
    # Add to relevant dictionary
    if num_sensors == 1:
        heat_one_houses[house_id] = data
    elif num_sensors == 2:
        heat_two_houses[house_id] = data
    elif num_sensors == 3:
        heat_three_houses[house_id] = data
    elif num_sensors == 4:
        heat_four_houses[house_id] = data
    elif num_sensors == 5:
        heat_five_houses[house_id] = data
        
print(f"Number of houses with 1 sensor: {len(heat_one_houses)}")
print(f"Number of houses with 2 sensors: {len(heat_two_houses)}")
print(f"Number of houses with 3 sensors: {len(heat_three_houses)}")
print(f"Number of houses with 4 sensors: {len(heat_four_houses)}")
print(f"Number of houses with 5 sensors: {len(heat_five_houses)}")

#%% Only for houses with 5 sensors.
rooms=['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]

# Initialize lists to hold the results
coldest_room_setpoint_differences = []
hottest_room_setpoint_differences = []
coldest_room_avg_temp_differences = []
hottest_room_avg_temp_differences = []

# Loop over the houses in the dictionary
for house_id, data in heat_five_houses.items():
    # Drop rows with missing values
    data = data.dropna(subset=rooms + ['Indoor_HeatSetpoint', 'Indoor_AverageTemperature'])

    # Compute the differences from the Indoor_HeatSetpoint and Indoor_AverageTemperature for all sensors
    setpoint_differences = data[rooms] - data['Indoor_HeatSetpoint'].values[:, None]
    avg_temp_differences = data[rooms] - data['Indoor_AverageTemperature'].values[:, None]

    # For each row, find the coldest and hottest room differences
    for index, row in setpoint_differences.iterrows():
        coldest_room = row.idxmin()
        hottest_room = row.idxmax()

        # Get the setpoint and average temperature differences for the coldest and hottest rooms
        coldest_room_setpoint_differences.append(setpoint_differences.loc[index, coldest_room])
        hottest_room_setpoint_differences.append(setpoint_differences.loc[index, hottest_room])
        coldest_room_avg_temp_differences.append(avg_temp_differences.loc[index, coldest_room])
        hottest_room_avg_temp_differences.append(avg_temp_differences.loc[index, hottest_room])

# Convert the lists to arrays
coldest_room_setpoint_differences = np.array(coldest_room_setpoint_differences)
hottest_room_setpoint_differences = np.array(hottest_room_setpoint_differences)
coldest_room_avg_temp_differences = np.array(coldest_room_avg_temp_differences)
hottest_room_avg_temp_differences = np.array(hottest_room_avg_temp_differences)

# Compute the average and standard deviation of the coldest and hottest room differences
print(f"Average setpoint difference for coldest room: {coldest_room_setpoint_differences.mean()}, with standard deviation: {coldest_room_setpoint_differences.std()}")
print(f"Average setpoint difference for hottest room: {hottest_room_setpoint_differences.mean()}, with standard deviation: {hottest_room_setpoint_differences.std()}")
print(f"Average average temperature difference for coldest room: {coldest_room_avg_temp_differences.mean()}, with standard deviation: {coldest_room_avg_temp_differences.std()}")
print(f"Average average temperature difference for hottest room: {hottest_room_avg_temp_differences.mean()}, with standard deviation: {hottest_room_avg_temp_differences.std()}")

#%%# for all houses
coldest_room_setpoint_differences = []
hottest_room_setpoint_differences = []
coldest_room_avg_temp_differences = []
hottest_room_avg_temp_differences = []

# Define list of house dictionaries and number of sensors in each
new_house_dictionaries = [heat_one_houses, heat_two_houses, heat_three_houses, heat_four_houses, heat_five_houses]
num_sensors = [1, 2, 3, 4, 5]

# Loop over the house dictionaries
for house_dict, num_sensor in zip(new_house_dictionaries, num_sensors):
    rooms = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, num_sensor+1)]

    # Loop over the houses in the dictionary
    for house_id, data in house_dict.items():
        # Drop rows with missing values
        data = data.dropna(subset=rooms + ['Indoor_HeatSetpoint', 'Indoor_AverageTemperature'])

        # Compute the differences from the Indoor_HeatSetpoint and Indoor_AverageTemperature for all sensors
        setpoint_differences = data[rooms] - data['Indoor_HeatSetpoint'].values[:, None]
        avg_temp_differences = data[rooms] - data['Indoor_AverageTemperature'].values[:, None]

        # For each row, find the coldest and hottest room differences
        for index, row in setpoint_differences.iterrows():
            coldest_room = row.idxmin()
            hottest_room = row.idxmax()

            # Get the setpoint and average temperature differences for the coldest and hottest rooms
            coldest_room_setpoint_differences.append(setpoint_differences.loc[index, coldest_room])
            hottest_room_setpoint_differences.append(setpoint_differences.loc[index, hottest_room])
            coldest_room_avg_temp_differences.append(avg_temp_differences.loc[index, coldest_room])
            hottest_room_avg_temp_differences.append(avg_temp_differences.loc[index, hottest_room])

# Convert the lists to arrays
coldest_room_setpoint_differences = np.array(coldest_room_setpoint_differences)
hottest_room_setpoint_differences = np.array(hottest_room_setpoint_differences)
coldest_room_avg_temp_differences = np.array(coldest_room_avg_temp_differences)
hottest_room_avg_temp_differences = np.array(hottest_room_avg_temp_differences)

# Compute the average and standard deviation of the coldest and hottest room differences
print(f"Average setpoint difference for coldest room: {coldest_room_setpoint_differences.mean()}, with standard deviation: {coldest_room_setpoint_differences.std()}")
print(f"Average setpoint difference for hottest room: {hottest_room_setpoint_differences.mean()}, with standard deviation: {hottest_room_setpoint_differences.std()}")
print(f"Average average temperature difference for coldest room: {coldest_room_avg_temp_differences.mean()}, with standard deviation: {coldest_room_avg_temp_differences.std()}")
print(f"Average average temperature difference for hottest room: {hottest_room_avg_temp_differences.mean()}, with standard deviation: {hottest_room_avg_temp_differences.std()}")
#%%

# Initialize a figure
fig, axs = plt.subplots(figsize=(15,10))

# Define a list of all house dictionaries
new_house_dictionaries = [heat_one_houses, heat_two_houses, heat_three_houses, heat_four_houses, heat_five_houses]

# Colors for different categories
colors = [(0,0,1,0.5), (1,0,0,0.5), (0,1,0,0.5), (1,0,1,0.5), (0,1,1,0.5)]

position = 0
positions = []
separator_positions = []

# Calculate and plot the deviations for each house dictionary
for i, houses in enumerate(new_house_dictionaries):
    deviations_houses = {}
    for house_id, data in houses.items():
        deviations = {}
        deviations['Thermostat_Temperature'] = data['Thermostat_Temperature'] - data['Indoor_HeatSetpoint']
        for sensor in range(1, i+2):  # The number of sensors corresponds to the dictionary
            sensor_column = f'RemoteSensor{sensor}_Temperature'
            if sensor_column in data.columns:
                deviations[sensor_column] = data[sensor_column] - data['Indoor_HeatSetpoint']

        deviations_df = pd.DataFrame(deviations)
        deviations_houses[house_id] = deviations_df

    # Concatenate all dataframes in deviations_houses into a single dataframe
    all_deviations = pd.concat(deviations_houses.values())

    # Create boxplot for this dictionary
    old_names = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, i+2)]
    # Create boxplot for this dictionary
    bplot = axs.boxplot([all_deviations[col].dropna() for col in old_names],
                        positions=[j+1+position for j in range(len(old_names))],
                        widths=0.6,
                        notch=True,  # Add notch
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[i], linewidth=2),  # Adjust line width here
                        medianprops=dict(color='black', linewidth=2),    # and here
                        whiskerprops=dict(linewidth=2),                  # and here
                        capprops=dict(linewidth=2),                     # and here
                        whis=1.5,  # Add whiskers
                        showfliers=False)
    for j in range(len(old_names)):
        positions.append(j+1+position)
    separator_positions.append(positions[-1] + 0.5)
    position += len(old_names) + 0.1

# Customize sub-plot
#axs.set_title('Temperature Deviation From Indoor Heat Setpoint', fontsize=26)
axs.set_xlabel('Sensor', fontsize=26)
axs.set_ylabel('Temperature Deviation (°F)', fontsize=26)
axs.axhline(0, color='gray', linestyle='--', linewidth=0.5)

# Set x-axis labels at the center of the boxes
x_ticks_labels = []
for i in range(5):
    x_ticks_labels += ['T'] + [f'{j}' for j in range(1, i+2)]
axs.set_xticks(positions)
axs.set_xticklabels(x_ticks_labels, fontsize=26)
axs.tick_params(axis='y', labelsize=26)  # Increase y-axis label size
axs.set_ylim(-13, None)  # Set lower limit to -10, upper limit remains unchanged

# Set y-axis ticks
axs.set_yticks([-10, -5, 0, 5, 10])

# Creating dashed lines between house categories
for pos in separator_positions[:-1]:
    axs.axvline(pos, color='gray', linestyle='--', linewidth=0.5)

y_max = axs.get_ylim()[1]  # Get the maximum value of the y-axis
axs.axhspan(0, y_max, facecolor='0.7', alpha=0.4)

#
# Creating dashed lines between house categories
for pos in separator_positions[:-1]:
    axs.axvline(pos, color='gray', linestyle='--', linewidth=0.5)



# Define house counts
house_counts = [350, 93, 285, 42, 75]

# Create custom legend
patches = [mpatches.Patch(color=colors[i], label=f'{house_counts[i]}', alpha=0.5) for i in range(5)]
legend = axs.legend(handles=patches, fontsize=22, title='Count', title_fontsize=22, ncol=3)


# Legend position can be changed by setting the bbox_to_anchor property
legend.set_bbox_to_anchor((0.45, 0.21))  # Place legend outside plot area, to the right

# Show the figure
plt.tight_layout()
plt.show()

