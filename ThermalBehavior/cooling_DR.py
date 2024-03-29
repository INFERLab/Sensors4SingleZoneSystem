#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:20:11 2023

@author: ozanbaris
This script is for the analysis for DR behavior of houses with five additional sensors. 
It extracts free floating periods that would be similar to a DR event and then analyze the thermal comfort impacts.

"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

for house_id, single_house_data in house_data.items():
    # Identify when the HVAC system is in cooling or heating mode
    single_house_data.set_index('time', inplace=True)

    single_house_data['Cooling_Mode'] = single_house_data['CoolingEquipmentStage1_RunTime'].notna()


    # Identify the periods of cooling and heating
    cooling_start = single_house_data.loc[single_house_data['Cooling_Mode']].index.min()
    cooling_end = single_house_data.loc[single_house_data['Cooling_Mode']].index.max()


    # Extract cooling and heating season data
    cooling_season_data = single_house_data.loc[cooling_start : cooling_end]


    # Store in dictionaries
    cooling_season_dict[house_id] = cooling_season_data





#%%
# Create empty dictionaries for each category
one_houses = {}
two_houses = {}
three_houses = {}
four_houses = {}
five_houses = {}

#for house_id, data in house_data.items():
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


for house_id, single_house_data in cooling_season_dict.items():
    single_house_data.reset_index(inplace=True)

# Prepare dictionary to store valid periods for each house and each sensor
valid_periods_dict = {}

# For each sensor, check if its temperature is more than 20 F above Indoor_CoolSetpoint
sensor_columns = [f'RemoteSensor{i}_Temperature' for i in range(1, 6)] + ['Thermostat_Temperature']

for house_id, single_house_data in five_houses.items():
    # Ensure data is sorted by time
    single_house_data.sort_values(by='time', inplace=True)

    # Initialize house entry in valid_periods_dict
    valid_periods_dict[house_id] = {}

    # Identify when HVAC is off (Cooling and Heating)
    single_house_data['HVAC_Off'] = single_house_data['CoolingEquipmentStage1_RunTime'].isna() & single_house_data['HeatingEquipmentStage1_RunTime'].isna()

    # Identify groups of rows where HVAC is off
    single_house_data['Off_Period'] = (single_house_data['HVAC_Off'] != single_house_data['HVAC_Off'].shift()).cumsum()

    # Filter periods that are at least 2 hours long and Outdoor_Temperature is higher than the sensor temperature
    # Also filter for times between 10 AM and 5 PM and where sensor temperature increased by at least 2 F

    valid_periods = single_house_data[single_house_data['HVAC_Off']].groupby('Off_Period').filter(
        lambda x: (len(x) >= 12)
                  and (x['time'].dt.hour >= 12).all()
                  and (x['time'].dt.hour < 18).all()
                  and all((x[sensor].max() - x[sensor].min()) >= 2 for sensor in sensor_columns))

    # Add to the dictionary
    valid_periods_dict[house_id] = valid_periods

#%%

# Then calculate and print the statistics:
def print_statistics(diff_list, name):
    print(f"Statistics for {name}:")
    print("----------------------")
    print("Mean:", np.mean(diff_list))
    print("Median:", np.median(diff_list))
    print("Standard deviation:", np.std(diff_list))
    print("Minimum:", np.min(diff_list))
    print("Maximum:", np.max(diff_list))
    print()
#%%
    
# Prepare dictionary to store average time for each sensor in each house
average_time_dict = {}

for house_id, data in valid_periods_dict.items():
    # Initialize house entry in average_time_dict
    average_time_dict[house_id] = {}



    for sensor in sensor_columns:
        # List to store times it takes for sensor to increase 2F
        times = []

        # Go through each valid period
        for _, period in data.groupby('Off_Period'):
            # Compute the sensor temperature at the beginning of the period
            start_temp = period[sensor].iat[0]

            # Compute the times where sensor temperature is 2F larger than beginning temperature
            times_over_2F = period[period[sensor] >= start_temp + 2]['time']

            # If there are such times, choose the earliest one
            if not times_over_2F.empty:
                earliest_time_over_2F = times_over_2F.iat[0]

                # Compute the difference between earliest time and the start time
                time_diff = (earliest_time_over_2F - period['time'].iat[0]).total_seconds() / 60  # Convert to minutes
                times.append(time_diff)

        # Compute average time for this sensor
        if times:  # Only compute the average if the list is not empty
            average_time_dict[house_id][sensor] = sum(times) / len(times)
        else:
            average_time_dict[house_id][sensor] = None
#%%
    
# Find the room with the lowest and highest time value in each house
lowest_time_rooms = {}
highest_time_rooms = {}
thermostat_times = {}

for house_id, sensors in average_time_dict.items():
    # Skip houses with no valid data
    if not sensors:
        continue

    # Remove sensors with None value
    sensors = {k: v for k, v in sensors.items() if v is not None}

    if sensors:
        # Find the sensor (room) with the lowest time
        lowest_time_room = min(sensors, key=sensors.get)
        lowest_time_rooms[house_id] = sensors[lowest_time_room]

        # Find the sensor (room) with the highest time
        highest_time_room = max(sensors, key=sensors.get)
        highest_time_rooms[house_id] = sensors[highest_time_room]

        # Get the thermostat time if available
        if 'Thermostat_Temperature' in sensors:
            thermostat_times[house_id] = sensors['Thermostat_Temperature']

# Convert to lists for convenience
lowest_time_values = list(lowest_time_rooms.values())
highest_time_values = list(highest_time_rooms.values())
thermostat_time_values = list(thermostat_times.values())

# Print statistics
print_statistics(lowest_time_values, "Lowest time rooms")
print_statistics(highest_time_values, "Highest time rooms")
print_statistics(thermostat_time_values, "Thermostat times")


#%%

# Compute the time difference from the room with lowest time to the room with highest time in each house
time_diffs = [highest_time_rooms[house_id] - lowest_time_rooms[house_id] for house_id in lowest_time_rooms.keys()]

# Print statistics for the differences
print_statistics(time_diffs, "Differences between highest and lowest time rooms")
#%%
# Prepare dictionary to store average temperature difference for each sensor in each house
average_temp_diff_dict = {}

for house_id, data in valid_periods_dict.items():
    # Initialize house entry in average_temp_diff_dict
    average_temp_diff_dict[house_id] = {}

    for sensor in sensor_columns:
        # List to store temperature differences for each period
        temp_diffs = []

        # Go through each valid period
        for _, period in data.groupby('Off_Period'):
            # Compute the sensor temperature at the beginning of the period
            start_temp = period[sensor].iat[0]

            # Compute the sensor temperature at the end of the period
            end_temp = period[sensor].iat[-1]

            # Compute the temperature difference
            temp_diff = end_temp - start_temp

            # Store the temperature difference for this period
            temp_diffs.append(temp_diff)

        # Compute average temperature difference for this sensor
        if temp_diffs:  # Only compute the average if the list is not empty
            average_temp_diff_dict[house_id][sensor] = sum(temp_diffs) / len(temp_diffs)
        else:
            average_temp_diff_dict[house_id][sensor] = None
   #%%         
# Prepare lists to store the temperature differences and thermostat temperatures
smallest_diff_values = []
largest_diff_values = []
thermostat_diffs = []
comfort_gaps = []

# Iterate over each house's data
for house_id, sensors in average_temp_diff_dict.items():
    # Skip houses with no valid data
    if not sensors:
        continue

    # Remove sensors with None value
    sensors = {k: v for k, v in sensors.items() if v is not None}

    if sensors:
        # Find the sensor (room) with the smallest temperature difference
        smallest_diff_room = min(sensors, key=sensors.get)
        smallest_diff_values.append(sensors[smallest_diff_room])

        # Find the sensor (room) with the largest temperature difference
        largest_diff_room = max(sensors, key=sensors.get)
        largest_diff_values.append(sensors[largest_diff_room])

        # Get the thermostat temperature difference if available
        if 'Thermostat_Temperature' in sensors:
            thermostat_diffs.append(sensors['Thermostat_Temperature'])

        # Compute the comfort gap for this house
        comfort_gap = sensors[largest_diff_room] - sensors[smallest_diff_room]
        comfort_gaps.append(comfort_gap)

# Print statistics
print_statistics(smallest_diff_values, "Smallest difference rooms")
print_statistics(largest_diff_values, "Largest difference rooms")
print_statistics(thermostat_diffs, "Thermostat differences")
print_statistics(comfort_gaps, "Comfort gaps")
