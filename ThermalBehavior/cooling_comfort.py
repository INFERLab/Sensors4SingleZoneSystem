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
from scipy.stats import iqr
import matplotlib.patches as mpatches
import numpy as np
import math

# File names
file_names = ['Mar_clean.nc','Apr_clean.nc', 'May_clean.nc', 'Jun_clean.nc', 'Jul_clean.nc', 'Aug_clean.nc', 'Sep_clean.nc','Oct_clean.nc','Nov_clean.nc',]


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
#%%# Prepare dictionaries to store cooling and heating season data for each house
cooling_season_dict = {}
no_cooling_houses = 0

for house_id, single_orig_data in house_data.items():
    # Identify when the HVAC system is in cooling or heating mode
    single_house_data = single_orig_data.copy()
    single_house_data.set_index('time', inplace=True)
    single_house_data['Cooling_Mode'] = single_house_data['CoolingEquipmentStage1_RunTime'].notna()
    single_house_data['Heating_Mode'] = single_house_data['HeatingEquipmentStage1_RunTime'].notna()

    # Create boolean Series with True when a switch from cooling to heating or from heating to cooling happens
    switches = single_house_data['Heating_Mode'].astype(int).diff().abs() == 1

    # Identify the start and end times of each period
    periods = single_house_data[switches].index

    # Extract heating and cooling data
    heating_data = single_house_data[single_house_data['Heating_Mode']]
    cooling_data = single_house_data[single_house_data['Cooling_Mode']]

    # Ensure that each period starts with heating and ends with cooling
    if not heating_data.empty and periods.size > 0 and periods[0] < heating_data.index.min():
        periods = periods[1:]
    if not cooling_data.empty and periods.size > 0 and periods[-1] > cooling_data.index.max():
        periods = periods[:-1]

    # Create tuples of each period's start and end times
    periods_tuples = list(zip(periods[::2], periods[1::2])) if periods.size > 0 else []

    # If there are transitions, identify the longest period (i.e., summer)
    if periods_tuples:
        summer_period = max(periods_tuples, key=lambda x: x[1] - x[0])
    else:
        # If there are no transitions, use the first and last instances of cooling as the summer period
        if not cooling_data.empty:  # If there is cooling data
            summer_period = (cooling_data.index.min(), cooling_data.index.max())
        else:
            # If there is no cooling data, skip this house and add to the counter of no cooling houses
            no_cooling_houses += 1
            continue

    # Extract summer data
    summer_data = single_house_data.loc[summer_period[0]:summer_period[1]]

    # Store in the dictionary
    cooling_season_dict[house_id] = summer_data

# Print the number of houses with no cooling
print(f"There are {no_cooling_houses} houses without any cooling.")



#%%
# Create empty dictionaries for each category
one_houses = {}
two_houses = {}
three_houses = {}
four_houses = {}
five_houses = {}


for house_id, data in cooling_season_dict.items():
    # Count the number of non-empty temperature sensors
    num_sensors = sum([1 for i in range(1, 6) if not np.isnan(data[f'RemoteSensor{i}_Temperature']).all()])
    
    # Add to relevant dictionary
    if num_sensors == 1:
        one_houses[house_id] = data
    elif num_sensors == 2:
        two_houses[house_id] = data
    elif num_sensors == 3:
        three_houses[house_id] = data
    elif num_sensors == 4:
        four_houses[house_id] = data
    elif num_sensors == 5:
        five_houses[house_id] = data
        
print(f"Number of houses with 1 sensor: {len(one_houses)}")
print(f"Number of houses with 2 sensors: {len(two_houses)}")
print(f"Number of houses with 3 sensors: {len(three_houses)}")
print(f"Number of houses with 4 sensors: {len(four_houses)}")
print(f"Number of houses with 5 sensors: {len(five_houses)}")

#%%
rooms=['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]

# Initialize lists to hold the results
coldest_room_setpoint_differences = []
hottest_room_setpoint_differences = []
coldest_room_avg_temp_differences = []
hottest_room_avg_temp_differences = []

# Loop over the houses in the dictionary
for house_id, data in five_houses.items():
    # Drop rows with missing values
    data = data.dropna(subset=rooms + ['Indoor_CoolSetpoint', 'Indoor_AverageTemperature'])

    # Compute the differences from the Indoor_HeatSetpoint and Indoor_AverageTemperature for all sensors
    setpoint_differences = data[rooms] - data['Indoor_CoolSetpoint'].values[:, None]
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

#%%# Initialize lists to hold the results
coldest_room_setpoint_differences = []
hottest_room_setpoint_differences = []
coldest_room_avg_temp_differences = []
hottest_room_avg_temp_differences = []

# Define list of house dictionaries and number of sensors in each
house_dictionaries = [one_houses, two_houses, three_houses, four_houses, five_houses]
num_sensors = [1, 2, 3, 4, 5]

# Loop over the house dictionaries
for house_dict, num_sensor in zip(house_dictionaries, num_sensors):
    rooms = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, num_sensor+1)]

    # Loop over the houses in the dictionary
    for house_id, data in house_dict.items():
        # Drop rows with missing values
        data = data.dropna(subset=rooms + ['Indoor_CoolSetpoint', 'Indoor_AverageTemperature'])

        # Compute the differences from the Indoor_HeatSetpoint and Indoor_AverageTemperature for all sensors
        setpoint_differences = data[rooms] - data['Indoor_CoolSetpoint'].values[:, None]
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
#%%#

# Initialize the names for the sensors and thermostat
old_names = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]
new_names = ['Thermostat'] + [f'Sensor{i}' for i in range(1, 6)]

# Compute the deviations for each sensor
deviations_five_houses = pd.DataFrame()
for house_id, data in five_houses.items():
    deviations = {}

    # Compute the deviation for the thermostat temperature
    deviations['Thermostat_Temperature'] = data['Thermostat_Temperature'] - data['Indoor_CoolSetpoint']

    # Compute the deviations for the remote sensor temperatures
    for i in range(1, 6):
        sensor_column = f'RemoteSensor{i}_Temperature'
        if sensor_column in data.columns:
            deviations[sensor_column] = data[sensor_column] - data['Indoor_CoolSetpoint']

    deviations_df = pd.DataFrame(deviations)
    deviations_five_houses = pd.concat([deviations_five_houses, deviations_df])

# Plot the deviations
plt.figure(figsize=(10,6))

# Box plot with desired properties
bplot = plt.boxplot([deviations_five_houses[col].dropna() for col in old_names],
                    positions=range(1, len(old_names) * 2, 2),
                    widths=0.6,
                    notch=True,   # Add notch
                    patch_artist=True,
                    boxprops=dict(facecolor=(0,0,1,0.5)),
                    medianprops=dict(color='black'),
                    whis=1.5,     # Add whiskers
                    showfliers=False)

# Customize plot
plt.title('Temperature Deviation From Indoor Cool Setpoint - Houses with Five Sensors')
plt.xlabel('Sensor')
plt.ylabel('Temperature Deviation (°F)')
plt.xticks(range(1, len(old_names) * 2, 2), new_names)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.show()


#%%

# Initialize a figure
fig, axs = plt.subplots(figsize=(15,10))

# Define a list of all house dictionaries
house_dictionaries = [one_houses, two_houses, three_houses, four_houses, five_houses]

# Colors for different categories
colors = [(0,0,1,0.5), (1,0,0,0.5), (0,1,0,0.5), (1,0,1,0.5), (0,1,1,0.5)]

position = 0
positions = []
separator_positions = []

# Calculate and plot the deviations for each house dictionary
for i, houses in enumerate(house_dictionaries):
    deviations_houses = {}
    for house_id, data in houses.items():
        deviations = {}
        deviations['Thermostat_Temperature'] = data['Thermostat_Temperature'] - data['Indoor_CoolSetpoint']
        for sensor in range(1, i+2):  # The number of sensors corresponds to the dictionary
            sensor_column = f'RemoteSensor{sensor}_Temperature'
            if sensor_column in data.columns:
                deviations[sensor_column] = data[sensor_column] - data['Indoor_CoolSetpoint']

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
#axs.set_title('Temperature Deviation From Indoor Cool Setpoint', fontsize=26)
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

# Creating dashed lines between house categories
for pos in separator_positions[:-1]:
    axs.axvline(pos, color='gray', linestyle='--', linewidth=0.5)

# Draw grey area from the minimum y-axis value to the zero level
y_min = axs.get_ylim()[0]  # Get the minimum value of the y-axis
axs.axhspan(y_min, 0, facecolor='0.7', alpha=0.4)
# Reset the y-axis limits to ensure shading covers the entire region
axs.set_ylim(y_min, axs.get_ylim()[1])
# Define house counts
house_counts = [387, 85, 308, 42, 65]

# Create custom legend
patches = [mpatches.Patch(color=colors[i], label=f'{house_counts[i]}', alpha=0.5) for i in range(5)]
legend = axs.legend(handles=patches, fontsize=22, title='Count', title_fontsize=22)

# Legend position can be changed by setting the bbox_to_anchor property
legend.set_bbox_to_anchor((0.67, 0.35))  # Place legend outside plot area, to the right

# Show the figure
plt.tight_layout()
plt.show()


#%%


# Initialize lists to hold the outlier values and corresponding outdoor temperatures
outlier_values = []
outdoor_temps = []

# Define temperature columns
temp_columns = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]

# Loop over the house dictionaries
for house_dict in [one_houses, two_houses, three_houses, four_houses, five_houses]:
    # Compute deviations
    deviations_houses = {}
    for house_id, data in house_dict.items():
        # Remove rows where the Indoor_CoolSetpoint is above 100
        data = data[data['Indoor_CoolSetpoint'] <= 100]
        
        deviations = {}
        deviations['Thermostat_Temperature'] = data['Thermostat_Temperature'] - data['Indoor_CoolSetpoint']
        for i, sensor in enumerate(temp_columns[1:], start=1):  # start indexing from 1
            sensor_column = f'RemoteSensor{i}_Temperature'
            if sensor_column in data.columns:
                deviations[sensor_column] = data[sensor_column] - data['Indoor_CoolSetpoint']

        deviations_df = pd.DataFrame(deviations)
        deviations_houses[house_id] = deviations_df

    # Now find the outliers in these deviations
    for house_id, data in deviations_houses.items():
        for col in temp_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = iqr(data[col].dropna())
                # Compute the whisker thresholds
                lower_threshold = Q1 - 1.5 * IQR
                upper_threshold = Q3 + 1.5 * IQR

                # Filter for outliers
                outliers = data[(data[col] < lower_threshold) | (data[col] > upper_threshold)]

                # Append the outliers and corresponding outdoor temperatures to the lists
                outlier_values += list(outliers[col])
                outdoor_temps += list(house_dict[house_id]['Outdoor_Temperature'].loc[outliers.index])

# Plot the outliers against their corresponding outdoor temperatures
plt.scatter(outdoor_temps, outlier_values, alpha=0.5)
plt.xlabel('Outdoor Temperature (°F)')
plt.ylabel('Outlier Deviation from Indoor Cool Setpoint (°F)')
plt.title('Outlier Values and their Corresponding Outdoor Temperatures')
plt.grid(True)
plt.show()

#%%
state_dict = {}

for house_dict in [one_houses, two_houses, three_houses, four_houses, five_houses]:
    for house_id, single_house_data in house_dict.items():
        # Get the cooling data from the cooling_season_dict
        # Save the state for each house
        state_values = single_house_data['State'].dropna().unique()
        assigned_state = 'Unknown'
        for state in state_values:
            if state != '':
                assigned_state = state
                break
        state_dict[house_id] = assigned_state


#%%

rooms = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]
def calculate_comfort_metrics(merged_df, rooms, x_range_step=0.5):
    all_R_values = {}
    comfort_metrics = {}

    for room in rooms:
        merged_df[f'{room}_diff'] = merged_df[room] - merged_df['Indoor_CoolSetpoint']
    
    max_deviations = [merged_df[f'{room}_diff'].max() for room in rooms]
    min_deviations = [merged_df[f'{room}_diff'].min() for room in rooms]
    abs_max_deviation = max(max(max_deviations), abs(min(min_deviations)))

    x_values = np.arange(-math.ceil(abs_max_deviation), math.ceil(abs_max_deviation) + 1, x_range_step)
    
    for room in rooms:
        R_values = []
        total_comf = 0
        comfort_2F = 0

        for x in x_values:
            Si_lower = merged_df[merged_df[f'{room}_diff'] <= x - 0.5]
            Si_upper = merged_df[merged_df[f'{room}_diff'] <= x]

            R = (len(Si_upper) - len(Si_lower)) / len(merged_df[f'{room}_diff'].dropna())
            R_values.append(R)
        
        all_R_values[room] = R_values

        total_comf = np.trapz(R_values, x_values)
        comfort_2F = np.trapz([r for x, r in zip(x_values, R_values) if x <= 0], x_values[(x_values <= 0)])
        comfort_metrics[room] = comfort_2F / total_comf if total_comf != 0 else 0

    return all_R_values, comfort_metrics



#%%
import itertools

# Define the list of rooms
rooms = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, 6)]

# Initialise the dictionary to hold results
five_houses_metrics = {}

# Loop through each house in the five_houses dictionary
for house_id, data in five_houses.items():

    # Compute the comfort metrics for this house using the predefined list of rooms
    all_R_values, comfort_metrics = calculate_comfort_metrics(data, rooms)

    # Save the comfort metrics for this house
    five_houses_metrics[house_id] = comfort_metrics
#%%
# Initialize a list to store the comfort metric differences
comfort_metric_diffs = []

# Loop through each house in the five_houses_metrics dictionary
for house_id, metrics in five_houses_metrics.items():
    # Calculate the difference between the thermostat comfort metric and the minimum room comfort metric
    diff = metrics['Thermostat_Temperature'] - min(metrics.values())
    
    # Append the difference to the list
    comfort_metric_diffs.append(diff)

# Convert the list to a numpy array
comfort_metric_diffs = np.array(comfort_metric_diffs)

# Calculate and print the average difference
average_diff = np.mean(comfort_metric_diffs)
print(f"Average comfort metric difference: {average_diff}")

# Calculate and print the standard deviation of the differences
std_dev_diff = np.std(comfort_metric_diffs)
print(f"Standard deviation of comfort metric differences: {std_dev_diff}")
#%%

# Define the print_statistics function
def print_statistics(diff_list, name):
    print(f"Statistics for {name}:")
    print("----------------------")
    print("Mean:", np.mean(diff_list))
    print("Median:", np.median(diff_list))
    print("Standard deviation:", np.std(diff_list))
    print("Minimum:", np.min(diff_list))
    print("Maximum:", np.max(diff_list))
    print()

# Initialize lists to store the comfort metrics for the thermostat, lowest room, and highest room
thermostat_metrics = []
lowest_room_metrics = []
highest_room_metrics = []

# Loop through each house in the five_houses_metrics dictionary
for house_id, metrics in five_houses_metrics.items():
    # Get the comfort metric for the thermostat and append it to the appropriate list
    thermostat_metrics.append(metrics['Thermostat_Temperature'])
    
    # Calculate the min and max room comfort metrics and append them to the appropriate lists
    min_val = min(metrics.values())
    max_val = max(metrics.values())
    lowest_room_metrics.append(min_val)
    highest_room_metrics.append(max_val)

# Use the print_statistics function to print the statistics for each array
print_statistics(thermostat_metrics, "Thermostat Metrics")
print_statistics(lowest_room_metrics, "Lowest Room Metrics")
print_statistics(highest_room_metrics, "Highest Room Metrics")










