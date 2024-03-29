#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:20:11 2023

@author: ozanbaris
This script computes the RK values using ecobee dataset. It uses balance point method. 

"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb, to_hex
from scipy.stats import linregress

# File names
file_names = ['Jan_clean.nc', 'Feb_clean.nc', 'Mar_clean.nc', 'Apr_clean.nc', 
              'May_clean.nc', 
              'Sep_clean.nc', 'Oct_clean.nc', 'Nov_clean.nc', 'Dec_clean.nc']

# Initialize an empty DataFrame
df_heat = pd.DataFrame()

# Load each file and append it to df_heat
for file_name in file_names:
    # Load the .nc file
    data = xr.open_dataset(file_name)

    # Convert it to a DataFrame
    temp_df_heat = data.to_dataframe()

    # Reset the index
    temp_df_heat = temp_df_heat.reset_index()

    # Append the data to df_heat
    df_heat = pd.concat([df_heat, temp_df_heat], ignore_index=True)


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

#%%# 
# Prepare dictionaries to store cooling and heating season data for each house
heating_season_dict = {}

for house_id, single_house_heat_data in house_heat_data.items():
    # Make a copy of the DataFrame to keep the original one intact
    single_house_heat_data_copy = single_house_heat_data.copy()

    # Set 'time' column as index
    single_house_heat_data_copy.set_index('time', inplace=True)

    # Identify when the HVAC system is in heating mode
    single_house_heat_data_copy['Heating_Mode'] = single_house_heat_data_copy['HeatingEquipmentStage1_RunTime'].notna()

    # Identify the periods of heating
    heating_start = single_house_heat_data_copy.loc[single_house_heat_data_copy['Heating_Mode']].index.min()
    heating_end = single_house_heat_data_copy.loc[single_house_heat_data_copy['Heating_Mode']].index.max()

    # Extract heating season data
    heating_season_data = single_house_heat_data.loc[(single_house_heat_data['time'] >= heating_start) & (single_house_heat_data['time'] <= heating_end)]

    # Store in dictionaries
    heating_season_dict[house_id] = heating_season_data


#%%
# Create empty dictionaries for each category
one_houses = {}
two_houses = {}
three_houses = {}
four_houses = {}
five_houses = {}

#for house_id, data in house_heat_data.items():
for house_id, data in heating_season_dict.items():
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

# Initialize counter
num_houses_with_non_nan_values = 0

for house_id, data in five_houses.items():
    # Check if the house has any non-NaN values in the specified column
    if data['HeatingEquipmentStage2_RunTime'].notna().any():
        num_houses_with_non_nan_values += 1

print(f"Number of houses with non-NaN values in 'HeatingEquipmentStage1_RunTime': {num_houses_with_non_nan_values}")



#%%

# Initialize dictionary to store results
night_data_dict = {}

sensor_columns = [f'RemoteSensor{i}_Temperature' for i in range(1, 6)] + ['Thermostat_Temperature']

for house_id, single_house_data in five_houses.items():
    single_house_data = single_house_data.copy()  # create a copy to avoid modifying the original dataframe

    # Sum up HeatingEquipmentStage1_RunTime and HeatingEquipmentStage2_RunTime
    single_house_data['TotalHeating_RunTime'] = single_house_data[['HeatingEquipmentStage1_RunTime', 'HeatingEquipmentStage2_RunTime']].sum(axis=1)

    # Create 'Night' column to identify nights
    single_house_data.loc[:, 'Night'] = single_house_data['time'].apply(lambda x: x.date() if 7 <= x.hour < 22 else (x - pd.Timedelta(days=1)).date() if x.hour >= 22 else x.date())
    
    # Filter out the rows where Cooling is on
    single_house_data = single_house_data[single_house_data['CoolingEquipmentStage1_RunTime'].isna()]

    # Filter out the rows which are not during the night
    single_house_data = single_house_data[(single_house_data['time'].dt.hour >= 22) | (single_house_data['time'].dt.hour < 7)]
    
    night_data_dict[house_id] = {}

    for sensor in sensor_columns:
        data_for_each_night = []  # list to store data for each night
        for night, night_data in single_house_data.groupby('Night'):
            # Calculate heating duty cycle
            total_night_duration_seconds = len(night_data) * 5 * 60  # number of rows times 5 minutes times 60 to get seconds
            heating_duty_cycle = night_data['TotalHeating_RunTime'].sum() / total_night_duration_seconds

            # Calculate average sensor temperature
            avg_sensor_temperature = night_data[sensor].mean()

            # Calculate average outdoor temperature
            avg_outdoor_temperature = night_data['Outdoor_Temperature'].mean()

            # Save results to list
            data_for_each_night.append({
                'Night': night,
                'avg_sensor_temperature': avg_sensor_temperature,
                'avg_outdoor_temperature': avg_outdoor_temperature,
                'heating_duty_cycle': heating_duty_cycle
            })

        # Convert list to DataFrame and save it to dictionary
        night_data_dict[house_id][sensor] = pd.DataFrame(data_for_each_night)

night_data_dict


#%%

for house_id, house_data in night_data_dict.items():
    for sensor, sensor_data in house_data.items():
        # Use pandas DataFrame.loc function to filter out rows where 'heating_duty_cycle' equals zero
        night_data_dict[house_id][sensor] = sensor_data.loc[sensor_data['heating_duty_cycle'] != 0]
#%%
# Select 5 random house-sensor couples
random.seed(0)  # Set the seed to ensure reproducibility
random_houses = random.sample(list(night_data_dict.keys()), 5)

fig, axs = plt.subplots(5, 2, figsize=(15, 20))  # Create 5 subplots with 2 columns

for i, house_id in enumerate(random_houses):
    # Choose one random sensor from the house
    random_sensor = random.choice(list(night_data_dict[house_id].keys()))
    sensor_data = night_data_dict[house_id][random_sensor]

    # First type of plot
    axs[i, 0].scatter(sensor_data['avg_outdoor_temperature'], sensor_data['heating_duty_cycle'])
    axs[i, 0].set_title(f'House ID: {house_id}, Sensor: {random_sensor}')
    axs[i, 0].set_xlabel('Average Outdoor Temperature')
    axs[i, 0].set_ylabel('Heating Duty Cycle')
    axs[i, 0].grid(True)

    # Second type of plot
    temp_diff = sensor_data['avg_sensor_temperature'] - sensor_data['avg_outdoor_temperature']
    axs[i, 1].scatter(temp_diff, sensor_data['heating_duty_cycle'])
    axs[i, 1].set_title(f'House ID: {house_id}, Sensor: {random_sensor}')
    axs[i, 1].set_xlabel('Average Sensor Temperature - Average Outdoor Temperature')
    axs[i, 1].set_ylabel('Heating Duty Cycle')
    axs[i, 1].grid(True)

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt

# List to hold the number of nights for each house-sensor couple
num_nights = []

for house_id, sensors in night_data_dict.items():
    for sensor, nights_data in sensors.items():
        num_nights.append(len(nights_data))

# Plotting
plt.figure(figsize=(10, 6))
plt.hist(num_nights, bins=50, edgecolor='black')
plt.title("Histogram of the Number of Nights for Each House-Sensor Couple")
plt.xlabel('Number of Nights')
plt.ylabel('Count')
plt.show()


#%%
night_data_dict_no_outliers = {}

for house_id, sensors in night_data_dict.items():
    night_data_dict_no_outliers[house_id] = {}
    for sensor, nights_data in sensors.items():
        # Compute mean and standard deviation for each column
        mean = nights_data.mean()
        std = nights_data.std()

        # Define the bounds for data that we want to keep
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std

        # Filter the data
        for column in nights_data.columns[1:]:
            nights_data = nights_data[(nights_data[column] >= lower_bound[column]) & 
                                      (nights_data[column] <= upper_bound[column])]
        
        # Update the data in the new dictionary
        night_data_dict_no_outliers[house_id][sensor] = nights_data


#%%

def line_to_dict(line, filename):
    """Take relevant parts of scipy stats line object"""
    slope = line.slope
    intercept = line.intercept
    r_value = line.rvalue
    p_value = line.pvalue
    stderr = line.stderr
    
    return {
            "filename" : filename, "slope" : slope, "intercept" : intercept,
            "r_value" : r_value, "p_value" : p_value, "stderr" : stderr
    }


from scipy.stats import linregress

regression_results = {}

for house_id, sensors in night_data_dict_no_outliers.items():
    regression_results[house_id] = {}
    for sensor, nights_data in sensors.items():
        x = nights_data['avg_sensor_temperature'] - nights_data['avg_outdoor_temperature']
        y = nights_data['heating_duty_cycle']
        
        # Check if the data is not empty
        if not x.empty and not y.empty:
            # Fit the line using linear regression
            line = linregress(x, y)
            
            # Save the results in the dictionary
            regression_results[house_id][sensor] = line_to_dict(line, f"{house_id}_{sensor}")



#%%

# Select 10 random house-sensor pairs
random_pairs = random.sample(list(regression_results.items()), 10)

# Loop over the selected pairs
for house_id, sensors in random_pairs:
    for sensor, line_params in sensors.items():
        # Retrieve the data with and without outliers
        data_with_outliers = night_data_dict[house_id][sensor]
        data_no_outliers = night_data_dict_no_outliers[house_id][sensor]

        # Calculate the x and y values for the scatter plots
        x_outliers = data_with_outliers['avg_sensor_temperature'] - data_with_outliers['avg_outdoor_temperature']
        y_outliers = data_with_outliers['heating_duty_cycle']

        x_no_outliers = data_no_outliers['avg_sensor_temperature'] - data_no_outliers['avg_outdoor_temperature']
        y_no_outliers = data_no_outliers['heating_duty_cycle']

        # Calculate the x and y values for the regression line
        x_line = np.linspace(min(x_outliers), max(x_outliers), 100)
        y_line = line_params['slope'] * x_line + line_params['intercept']

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(x_outliers, y_outliers, color='red', label='With Outliers')
        plt.scatter(x_no_outliers, y_no_outliers, color='blue', label='No Outliers')
        plt.plot(x_line, y_line, color='green', label='Regression Line')
        
        # Set the title and labels
        plt.title(f'House {house_id} - Sensor {sensor}')
        plt.xlabel('Avg Sensor Temp - Avg Outdoor Temp')
        plt.ylabel('Heating Duty Cycle')
        plt.legend()

        # Display the plot
        plt.show()

#%%
regression_results_dict = {}
night_data_dict_no_outliers_v2 = {}

for house_id, house_sensors in night_data_dict.items():
    night_data_dict_no_outliers_v2[house_id] = {}
    for sensor, data in house_sensors.items():
        night_data_dict_no_outliers_v2[house_id][sensor] = data.copy()
        x = data['avg_sensor_temperature'] - data['avg_outdoor_temperature']
        y = data['heating_duty_cycle']
        
        # Fit the model with all data
        line = linregress(x, y)
        predictions = line.slope * x + line.intercept
        
        # Calculate the residuals
        residuals = y - predictions
        
        # Find the points that are within 2 standard deviations of the mean residual
        non_outliers = np.abs(residuals - np.mean(residuals)) < 2 * np.std(residuals)
        
        # Keep only the non-outlier data
        night_data_dict_no_outliers_v2[house_id][sensor] = data[non_outliers]

        # Re-fit the model with non-outlier data if there is more than one point remaining
        if len(night_data_dict_no_outliers_v2[house_id][sensor]) > 1:
            x = night_data_dict_no_outliers_v2[house_id][sensor]['avg_sensor_temperature'] - night_data_dict_no_outliers_v2[house_id][sensor]['avg_outdoor_temperature']
            y = night_data_dict_no_outliers_v2[house_id][sensor]['heating_duty_cycle']
            line = linregress(x, y)

            regression_results_dict[(house_id, sensor)] = line_to_dict(line, f'{house_id}_{sensor}')

# Plotting
random_samples = random.sample(list(regression_results_dict.keys()), 10)

for house_sensor in random_samples:
    house_id, sensor = house_sensor
    x = night_data_dict_no_outliers_v2[house_id][sensor]['avg_sensor_temperature'] - night_data_dict_no_outliers_v2[house_id][sensor]['avg_outdoor_temperature']
    y = night_data_dict_no_outliers_v2[house_id][sensor]['heating_duty_cycle']
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data after removing outliers')
    plt.plot(x, regression_results_dict[house_sensor]['slope'] * x + regression_results_dict[house_sensor]['intercept'], color='red', label='Fitted line')
    plt.xlabel('avg_sensor_temperature - avg_outdoor_temperature')
    plt.ylabel('heating_duty_cycle')
    plt.legend()
    plt.title(f'House ID: {house_id}, Sensor: {sensor}')
    plt.show()



#%%
RK_values = {}

for house_id, sensors in regression_results.items():
    RK_values[house_id] = {}
    for sensor, data in sensors.items():
        if house_id not in RK_values:
            RK_values[house_id] = {}
        RK_values[house_id][sensor] = 1/data['slope']
# Define the sensor names
sensor_names = [f'RemoteSensor{i}_Temperature' for i in range(1, 6)] + ['Thermostat_Temperature']


#%%
import numpy as np
import matplotlib.pyplot as plt

# Plotting the histogram of original RK values
original_RK = [val for subdict in RK_values.values() for val in subdict.values() if val >= 0 and isinstance(val, np.float64)]
plt.figure(figsize=(10, 6))
plt.hist(original_RK, bins=50, edgecolor='black')
plt.title('Histogram of Original RK Values')
plt.xlabel('RK Value')
plt.ylabel('Frequency')
plt.show()

# Filter by r_value first
filtered_by_r_value = {}

for house_id, sensors in regression_results.items():
    filtered_by_r_value[house_id] = {}
    for sensor, data in sensors.items():
        r_value = data['r_value']
        rk_value = RK_values[house_id][sensor]
        if r_value >= 0.7 and rk_value >= 0:
            filtered_by_r_value[house_id][sensor] = rk_value

# Plotting the histogram of RK values after r_value filtering
r_value_filtered_RK = [val for subdict in filtered_by_r_value.values() for val in subdict.values() if isinstance(val, np.float64)]
plt.figure(figsize=(10, 6))
plt.hist(r_value_filtered_RK, bins=50, edgecolor='black')
plt.title('Histogram of RK Values After r_value Filtering')
plt.xlabel('RK Value')
plt.ylabel('Frequency')
plt.show()

# Compute global mean and standard deviation
mean_rk = np.mean(r_value_filtered_RK)
std_rk = np.std(r_value_filtered_RK)

# Filter by standard deviation
std_RK_values = {}

for house_id, sensors in filtered_by_r_value.items():
    for sensor, rk_value in sensors.items():
        if abs(rk_value - mean_rk) <= 2*std_rk:
            # If the house_id does not exist in the std_RK_values, create an empty dict for it
            if house_id not in std_RK_values:
                std_RK_values[house_id] = {}
            # Save the sensor and its RK value
            std_RK_values[house_id][sensor] = rk_value

# Plotting the histogram of RK values after standard deviation filtering
std_filtered_RK = [val for subdict in std_RK_values.values() for val in subdict.values() if isinstance(val, np.float64)]
plt.figure(figsize=(10, 6))
plt.hist(std_filtered_RK, bins=50, edgecolor='black')
plt.title('Histogram of RK Values After Filtering')
plt.xlabel('RK Value (F)')
plt.ylabel('Frequency')
plt.show()
#%%
import pandas as pd
import numpy as np

# Read the CSV files into DataFrames
final_df_RC = pd.read_csv('BuildSys/std_RK_values.csv', index_col=0)


def dataframe_to_dict(df):
    """Converts a DataFrame to a nested dictionary."""
    nested_dict = df.transpose().to_dict()
    for house_id, sensors in nested_dict.items():
        for sensor, value in sensors.items():
            if np.isnan(value):
                nested_dict[house_id][sensor] = None
            else:
                nested_dict[house_id][sensor] = float(value)
    return nested_dict

# Convert the DataFrames back to dictionary form
std_RK_values = dataframe_to_dict(final_df_RC)

#%%
# Extract the house_ids from the final_RC_values dictionary
house_ids_from_final_RC = list(std_RK_values.keys())

# Initialize the state_dict
state_dict = {}

for house_id in house_ids_from_final_RC:
    # Filter the dataframe for the current house_id
    single_house_data = df_heat[df_heat['id'] == house_id]
    
    # Get unique non-na states for the current house
    state_values = single_house_data['State'].dropna().unique()
    
    # Initialize the state for the current house as 'Unknown'
    assigned_state = 'Unknown'
    for state in state_values:
        if state != '':
            assigned_state = state
            break
    
    # Add the state to the state_dict
    state_dict[house_id] = assigned_state

print(state_dict)

#%%


# Function to darken a color
def darken(color, amount=0.1):
    # Convert color to RGB
    rgb = to_rgb(color)
    
    # Reduce each of the RGB values
    rgb = [x * (1 - amount) for x in rgb]
    
    # Convert RGB back to hex
    color = to_hex(rgb)
    
    return color

# Darken each color in state_colors
state_colors = {'TX': 'red', 'CA': 'orange', 'IL': 'blue', 'NY': 'green'}
state_colors_dark = {state: darken(color, 0.1) for state, color in state_colors.items()}

state_dict = {}


for house_id, single_house_data in five_houses.items():
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
# Darken each color in state_colors
state_colors = {'TX': 'red', 'CA': 'orange', 'IL': 'blue', 'NY': 'green'}
state_colors_dark = {state: darken(color, 0.1) for state, color in state_colors.items()}

def plot_data(df, var_name, title, unit, thermostat_df, filename='plot.pdf'):
    # Set the overall font size to 26
    plt.rcParams.update({'font.size': 26})

    # Create a gridspec
    gs = plt.GridSpec(2, 1, height_ratios=[1, 5], hspace=0)

    # Create figure
    fig = plt.figure(figsize=(15, 15))

    # Map house_id to states and add as a new column
    df['mapped_state'] = df['house_id'].map(state_dict)
    

    # Get the ordered list of unique house_ids from df
    ordered_house_ids = df['house_id'].unique()

    # Convert house_id in df into a categorical variable with its current order
    df['house_id'] = pd.Categorical(df['house_id'], categories=ordered_house_ids, ordered=True)

    # Create the histogram in the top subplot
    ax0 = plt.subplot(gs[0])
    # Plot histograms for each state
    sns.histplot(data=df, x=var_name, hue='mapped_state', palette=state_colors_dark, multiple="stack", bins=50, edgecolor=".3", linewidth=.5, ax=ax0, legend=False)
    ax0.set_ylabel('Count')  # Update y-axis label
    ax0.set_xlabel('')  # Remove x-axis label

    # Create the boxplot in the bottom subplot
    ax1 = plt.subplot(gs[1], sharex=ax0)  # share x-axis with the histogram
    bp = sns.boxplot(y='house_id', x=var_name, data=df, hue='mapped_state', palette=state_colors, dodge=False, ax=ax1)  # save the axis to bp
    # Filter out house_ids not in df
    thermostat_df = thermostat_df[thermostat_df['house_id'].isin(ordered_house_ids)]

    # Convert house_id in thermostat_df into a categorical variable with the same order as in df
    thermostat_df['house_id'] = pd.Categorical(thermostat_df['house_id'], categories=ordered_house_ids, ordered=True)

    # Now, you can sort the thermostat_df by 'house_id' in the same order as in df:
    thermostat_df = thermostat_df.sort_values('house_id')
    
    # You can add a rank column to thermostat_df, just like in df
    thermostat_df['rank'] = thermostat_df['house_id'].cat.codes

    # Plot markers over the boxplot
    ax1.scatter(thermostat_df['Thermostat_Temperature'], thermostat_df['rank'], color='cyan', marker="D", s=100)


    # Customize labels and legend
    ax1.set_xlabel(f'{title} Value ({unit})')  # Update x-axis label
    ax1.set_ylabel('House Rank')  # Remove y-axis label
    ax1.legend(title='State', bbox_to_anchor=(1, 0), loc='lower right')  # Place legend at the lower right end
    # Customize ytick labels
    yticks_locs = ax1.get_yticks()  # get current ytick locations
    new_yticks = [i+1 if i%10==0 else '' for i in range(len(yticks_locs))]  # show every 10th house number, starting from 1
    new_yticks[0] = ''  # Remove the tick on the first value of the House Number axis
    ax1.set_yticks(yticks_locs)  # set new ytick locations
    ax1.set_yticklabels(new_yticks)  # set new ytick labels
    
    plt.tight_layout()  # Make sure everything fits
    plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def compute_boxplot_data(final_values, sensor_names, var_name, state_dict):
    # Convert the final_values dict to a DataFrame
    df_values = pd.DataFrame(final_values).T
    df_values.reset_index(inplace=True)
    df_values.rename(columns={'index': 'house_id'}, inplace=True)

    # Melt the DataFrame to a long format
    boxplot_data = df_values.melt(id_vars='house_id', var_name='sensor', value_name=var_name)
    boxplot_data.dropna(subset=[var_name], inplace=True)
    
    # Sort the houses by the maximum value of var_name
    sorted_houses = boxplot_data.groupby("house_id")[var_name].max().sort_values(ascending=False).index
    boxplot_data["house_id"] = boxplot_data["house_id"].astype("category")
    boxplot_data["house_id"].cat.set_categories(sorted_houses, inplace=True)
    
    # Map the house_id to the state
    boxplot_data['state'] = boxplot_data['house_id'].map(state_dict)
    
    # Set the font size for the plot
    plt.rcParams.update({'font.size': 26})
    
    # Compute the range of var_name for each house
    var_range = boxplot_data.groupby("house_id")[var_name].max() - boxplot_data.groupby("house_id")[var_name].min()
    
    # Rank the houses by the range of var_name
    sorted_houses = var_range.sort_values(ascending=True).index
    rank_mapping = {house_id: i+1 for i, house_id in enumerate(sorted_houses)}
    
    # Assign ranks, giving a default value (max rank + 1) to houses not present in rank_mapping
    max_rank = len(rank_mapping)
    boxplot_data['rank'] = boxplot_data['house_id'].map(lambda x: rank_mapping.get(x, max_rank+1))
    
    # Convert rank to integer
    boxplot_data['rank'] = boxplot_data['rank'].astype(int)
    
    # Sort the data by the rank
    boxplot_data.sort_values('rank', inplace=True)

    # Create the thermostat_data DataFrame
# Create the thermostat_data DataFrame
    thermostat_data = pd.DataFrame({house_id: [final_values[house_id].get('Thermostat_Temperature', np.nan)] for house_id in final_values}).T
    thermostat_data.reset_index(inplace=True)
    thermostat_data.rename(columns={'index': 'house_id', 0: 'Thermostat_Temperature'}, inplace=True)

    return boxplot_data, thermostat_data


boxplot_data_RK, thermostat_data_RK=compute_boxplot_data(std_RK_values,sensor_names,'RK', state_dict)
# Calling the function
plot_data(boxplot_data_RK, 'RK', 'RK', 'F', thermostat_data_RK, filename="RK_heating.pdf")




 #%%
def compute_boxplot_data(std_values_dict, var_name, state_dict):

    df_values = pd.DataFrame(std_values_dict).T
    df_values.reset_index(inplace=True)
    df_values.rename(columns={'index': 'house_id'}, inplace=True)

    # Skip error filtering, since it's already done
    df_values_filtered = df_values.copy()
    df_values_filtered.dropna(how='all', subset=df_values_filtered.columns[1:], inplace=True)
    df_values_filtered.index = range(1, len(df_values_filtered) + 1)
    
    boxplot_data = df_values_filtered.melt(id_vars='house_id', var_name='sensor', value_name=var_name)
    boxplot_data.dropna(subset=[var_name], inplace=True)
    
    sorted_houses = boxplot_data.groupby("house_id")[var_name].max().sort_values(ascending=False).index
    boxplot_data["house_id"] = boxplot_data["house_id"].astype("category")
    boxplot_data["house_id"].cat.set_categories(sorted_houses, inplace=True)
    boxplot_data['state'] = boxplot_data['house_id'].map(state_dict)
    
    plt.rcParams.update({'font.size': 26})
    var_range = boxplot_data.groupby("house_id")[var_name].max() - boxplot_data.groupby("house_id")[var_name].min()
    sorted_houses = var_range.sort_values(ascending=True).index
    rank_mapping = {house_id: i+1 for i, house_id in enumerate(sorted_houses)}
    boxplot_data["rank"] = boxplot_data["house_id"].map(rank_mapping)
    boxplot_data.sort_values('rank', inplace=True)

    return boxplot_data

boxplot_data_RK = compute_boxplot_data(std_RK_values, 'RK', state_dict)


#boxplot_data_RC = compute_boxplot_data(five_houses, sensor_names, errors, 'RK', state_dict, RK_values, RC_values_sensors_no_outliers)
#boxplot_data_RQ = compute_boxplot_data(five_houses, sensor_names, errors, 'RQ', state_dict, RQ_values, RQ_values_sensors_no_outliers)
plot_data(boxplot_data_RK, 'RK', 'RK', 'F')
#plot_data(boxplot_data_RQ, 'RQ', 'RQ', 'F')
  #%%
def compute_statistics(boxplot_data, value_name):

    # Average value for all houses combined with its standard deviation
    average_all = boxplot_data[value_name].mean()
    std_dev_all = boxplot_data[value_name].std()
    print(f"Average {value_name} value for all houses: {average_all}")
    print(f"Standard deviation of {value_name} values for all houses: {std_dev_all}")

    # Maximum differences in each house and statistical results of those differences
    boxplot_data['max_difference'] = boxplot_data.groupby('house_id')[value_name].transform(lambda x: x.max() - x.min())
    difference_counts = boxplot_data['max_difference'].value_counts()

    # Compute probabilities for differences
    for diff in [5, 10, 15]:
        probability = (boxplot_data['max_difference'] >= diff).mean()
        print(f"\nProbability of having an {value_name} value in the same house {diff} more than the other {value_name} value: {probability}")

    # Compute likelihood of maximum difference within house
    sorted_diff = np.sort(boxplot_data['max_difference'].unique())
    median_diff = sorted_diff[len(sorted_diff) // 2]

    print(f"\nMore than 50% of the time, there will be a difference of at least {median_diff} in the {value_name} values of the sensors in the same house.")

    # Calculating probabilities of getting a value for a sensor that is x% more than the minimum value for a sensor in that house.
    for perc in [50, 75, 100, 150, 200]:
        boxplot_data[f'min_plus_{perc}'] = boxplot_data.groupby('house_id')[value_name].transform(lambda x: x.min() * (1 + perc / 100.0))
        probability = (boxplot_data[value_name] >= boxplot_data[f'min_plus_{perc}']).mean()
        print(f"\nProbability of getting an {value_name} value for a sensor that is {perc}% more than the minimum {value_name} value for a sensor in that house: {probability}")

    unique_houses = boxplot_data["house_id"].unique()

    diff_percentiles = []
    for house in unique_houses:
        house_values = boxplot_data[boxplot_data["house_id"] == house][value_name]
        min_val = house_values.min()
        max_val = house_values.max()
        diff_percent = ((max_val - min_val) / min_val) * 100
        diff_percentiles.append(diff_percent)

    upper_percentiles = [100 - p for p in [25, 50, 75]]
    upper_diffs = [np.percentile(diff_percentiles, p) for p in upper_percentiles]

    for p, diff in zip(upper_percentiles, upper_diffs):
        print(f"With {100-p}% probability, the percentile difference from the minimum to maximum {value_name} value in the same house is more than approximately {diff}%.")

compute_statistics(boxplot_data_RK, 'RK')

    
    
  #%%
def compute_percent_identification(dataset, std_values_dict, value_name):
    total_data = len(dataset) * 6
    non_nan_couples = 0

    # Calculate the number of house-sensor couples that are non-NaN for the given values
    for house_id, house_data in std_values_dict.items():
        for sensor, sensor_value in house_data.items():
            if not np.isnan(sensor_value):
                non_nan_couples += 1

    # Compute the percentage of identification
    percentage_identification = (non_nan_couples / total_data) * 100

    # Store everything in a dictionary
    results_dict = {
        'total_data': total_data,
        'non_nan_couples': non_nan_couples,
        'percentage_identification': percentage_identification
    }

    print(f"For {value_name} values:")
    print(f"Total data: {results_dict['total_data']}")
    print(f"Number of non-NaN house-sensor couples: {results_dict['non_nan_couples']}")
    print(f"Percentage of identification: {results_dict['percentage_identification']}%")

    return results_dict

rk_results = compute_percent_identification(five_houses, std_RK_values, 'RK')
  #%%

def dict_to_csv(dictionary, csv_filename):
    df = pd.DataFrame(dictionary).transpose()
    df.to_csv(csv_filename)


dict_to_csv(std_RK_values, 'final_std_RK_values.csv')

#%%
import seaborn as sns
state_dict = {}

# Set overall font size to 24
plt.rcParams.update({'font.size': 24})

for house_id, single_house_data in five_houses.items():
    # Get the cooling data from the cooling_season_dict
    # Save the state for each house
    state_values = single_house_data['State'].dropna().unique()
    assigned_state = 'Unknown'
    for state in state_values:
        if state != '':
            assigned_state = state
            break
    state_dict[house_id] = assigned_state

state_colors = {'TX': 'red', 'CA': 'orange', 'IL': 'blue', 'NY': 'green'}

# Creating a Pandas DataFrame to store RK values along with their house and sensor identifiers
boxplot_data = pd.DataFrame(columns=['house_id', 'sensor', 'RK', 'state'])

for house_id, sensors in std_RK_values.items():
    for sensor, rk_value in sensors.items():
        state = state_dict.get(house_id, 'Unknown')  # Get state of the house, if unknown set it to 'Unknown'
        boxplot_data = boxplot_data.append({'house_id': house_id, 'sensor': sensor, 'RK': rk_value, 'state': state}, ignore_index=True)

# Order the houses by their maximum RK value
house_max = boxplot_data.groupby('house_id')['RK'].max()
sorted_houses = house_max.sort_values(ascending=False).index.tolist()

plt.figure(figsize=(15, 10))

# Create boxplot with hue for states
boxplot = sns.boxplot(data=boxplot_data, x='house_id', y='RK', hue='state', palette=state_colors, order=sorted_houses, dodge=False)

# Customize x-axis labels
num_houses = len(sorted_houses)
xticklabels = [i+1 for i in range(num_houses)]  # sequential numbering from 1 to n
xticks = range(num_houses)

# Make every nth label visible, rest invisible
n = 10  # adjust this to control how many labels are displayed
visible_labels = [label if (i % n == 0) else '' for i, label in enumerate(xticklabels)]
plt.xticks(ticks=xticks, labels=visible_labels)

plt.title('Boxplot of RK Values for Each House')
plt.xlabel('House Number')
plt.ylabel('RK Value (F)')
plt.legend(title='State', bbox_to_anchor=(1, 1))  # Place legend outside the plot

plt.show()
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Set overall font size to 24
plt.rcParams.update({'font.size': 38})

plt.figure(figsize=(15, 10))

# Create histogram
sns.histplot(data=boxplot_data, x='RK', hue='state', palette=state_colors, kde=True,bins=50)

# Set the y-axis label to integers
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.title('Histogram of RK Values')
plt.xlabel('RK Value (F)')
plt.ylabel('Count')

plt.show()

plt.figure(figsize=(15, 10))

# Create stacked histogram
sns.histplot(data=boxplot_data, x='RK', hue='state', palette=state_colors, multiple="stack", bins=50, edgecolor=".3", linewidth=.5)

# Set the y-axis label to integers
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.title('Histogram of RK Values')
plt.xlabel('RK Value (F)')
plt.ylabel('Count')

plt.show()


#%%Plotting the succesful houses

# First, we need to get the list of top 10 houses. 
# To do this, we first create a dictionary where the keys are house IDs and the values are the max r_value for that house
house_max_r_values = {house_id: max([data['r_value'] for data in sensors.values()]) for house_id, sensors in regression_results.items()}
# Then we sort this dictionary by r_value and take the top 10
top_10_houses = sorted(house_max_r_values.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_houses = [house_id for house_id, _ in top_10_houses]

# Then we plot the sensors for each of these houses
for house_id in top_10_houses:
    # Get the sensors for this house
    sensors = regression_results[house_id]
    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    # Flatten the axs array
    axs = axs.ravel()
    # For each sensor
    for idx, (sensor_id, data) in enumerate(sensors.items()):
        # Get the x and y data
        x = night_data_dict_no_outliers[house_id][sensor_id]['avg_sensor_temperature'] - night_data_dict_no_outliers[house_id][sensor_id]['avg_outdoor_temperature']
        y = night_data_dict_no_outliers[house_id][sensor_id]['heating_duty_cycle']

        # Compute the line points
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = data['slope'] * line_x + data['intercept']

        # Plot the data and the line
        axs[idx].scatter(x, y, label='Data')
        axs[idx].plot(line_x, line_y, color='red', label='Fitted line')
        
        # Create the text for the annotation box
        annotation_text = f"r_value: {data['r_value']:.3f}\np_value: {data['p_value']:.3f}\nstderr: {data['stderr']:.3f}"
        axs[idx].annotate(annotation_text, xy=(0.02, 0.75), xycoords='axes fraction', bbox=dict(facecolor='white', alpha=0.8))

        # Set labels and title
        axs[idx].set_ylabel('Heating Duty Cycle')
        axs[idx].set_xlabel('Indoor Temperature - Outdoor Temperature (F)')
        axs[idx].set_title(f"{sensor_id}")
        axs[idx].legend()
        axs[idx].grid(True)
    
    # Remove any unused subplots
    for idx in range(len(sensors), 6):
        fig.delaxes(axs[idx])

    # Show the figure
    plt.tight_layout()
    plt.show()



#%% 
# Calculate number of non-nan values
original_RK_non_nan_count = len(original_RK)
filtered_by_r_value_non_nan_count = len(r_value_filtered_RK)
std_RK_values_non_nan_count = len(std_filtered_RK)

# Save the counts in a dictionary
results = {
    'original_RK': original_RK_non_nan_count,
    'filtered_by_r_value': filtered_by_r_value_non_nan_count,
    'std_RK_values': std_RK_values_non_nan_count
}

print(results)
#%% 
# Average RK value for all houses combined with its standard deviation
average_RK_all = boxplot_data['RK'].mean()
std_dev_RK_all = boxplot_data['RK'].std()
print(f"Average RK value for all houses: {average_RK_all}")
print(f"Standard deviation of RK values for all houses: {std_dev_RK_all}")

# Maximum differences in each house and statistical results of those differences
boxplot_data['max_difference'] = boxplot_data.groupby('house_id')['RK'].transform(lambda x: x.max() - x.min())
difference_counts = boxplot_data['max_difference'].value_counts()

print("\nDistribution of maximum differences in RK values within each house:")
print(difference_counts)

# Compute probabilities for differences of 5, 10, 15
for diff in [5, 10, 15]:
    probability = (boxplot_data['max_difference'] >= diff).mean()
    print(f"\nProbability of having an RK value in the same house {diff} more than the other RK value: {probability}")

# Compute likelihood of maximum difference within house
sorted_diff = np.sort(boxplot_data['max_difference'].unique())
median_diff = sorted_diff[len(sorted_diff) // 2]

print(f"\nMore than 50% of the time, there will be a difference of at least {median_diff} in the RK values of the sensors in the same house.")

# Calculating probabilities of getting an RK value for a sensor that is x% more than the minimum RK value for a sensor in that house.
for perc in [50, 75, 100, 150, 200]:
    boxplot_data[f'min_plus_{perc}'] = boxplot_data.groupby('house_id')['RK'].transform(lambda x: x.min() * (1 + perc / 100.0))
    probability = (boxplot_data['RK'] >= boxplot_data[f'min_plus_{perc}']).mean()
    print(f"\nProbability of getting an RK value for a sensor that is {perc}% more than the minimum RK value for a sensor in that house: {probability}")

unique_houses = boxplot_data["house_id"].unique()

diff_percentiles = []


for house in unique_houses:
    house_RK_values = boxplot_data[boxplot_data["house_id"] == house]["RK"]
    min_RK = house_RK_values.min()
    max_RK = house_RK_values.max()
    diff_percent = ((max_RK - min_RK) / min_RK) * 100
    diff_percentiles.append(diff_percent)

upper_percentiles = [100 - p for p in [25, 50, 75]]
upper_diffs = [np.percentile(diff_percentiles, p) for p in upper_percentiles]

for p, diff in zip(upper_percentiles, upper_diffs):
    print(f"With {100-p}% probability, the percentile difference from the minimum to maximum RK value in the same house is more than approximately {diff}%.")

#%% 
# Average RK value for all houses combined with its standard deviation
average_RK_all = boxplot_data['RK'].mean()
std_dev_RK_all = boxplot_data['RK'].std()
results['average_RK_all'] = average_RK_all
results['std_dev_RK_all'] = std_dev_RK_all

# Maximum differences in each house and statistical results of those differences
boxplot_data['max_difference'] = boxplot_data.groupby('house_id')['RK'].transform(lambda x: x.max() - x.min())
difference_counts = boxplot_data['max_difference'].value_counts()
results['max_difference_counts'] = difference_counts

# Compute probabilities for differences of 5, 10, 15
for diff in [5, 10, 15]:
    probability = (boxplot_data['max_difference'] >= diff).mean()
    results[f'prob_difference_{diff}'] = probability

# Compute likelihood of maximum difference within house
sorted_diff = np.sort(boxplot_data['max_difference'].unique())
median_diff = sorted_diff[len(sorted_diff) // 2]
results['median_difference'] = median_diff

# Calculating probabilities of getting an RK value for a sensor that is x% more than the minimum RK value for a sensor in that house.
for perc in [50, 75, 100, 150, 200]:
    boxplot_data[f'min_plus_{perc}'] = boxplot_data.groupby('house_id')['RK'].transform(lambda x: x.min() * (1 + perc / 100.0))
    probability = (boxplot_data['RK'] >= boxplot_data[f'min_plus_{perc}']).mean()
    results[f'prob_perc_{perc}'] = probability

unique_houses = boxplot_data["house_id"].unique()

diff_percentiles = []
for house in unique_houses:
    house_RK_values = boxplot_data[boxplot_data["house_id"] == house]["RK"]
    min_RK = house_RK_values.min()
    max_RK = house_RK_values.max()
    diff_percent = ((max_RK - min_RK) / min_RK) * 100
    diff_percentiles.append(diff_percent)
results['diff_percentiles'] = diff_percentiles

upper_percentiles = [100 - p for p in [25, 50, 75]]
upper_diffs = [np.percentile(diff_percentiles, p) for p in upper_percentiles]
results['upper_percentiles'] = upper_percentiles
results['upper_diffs'] = upper_diffs
#%% 
import csv

# Define file paths
rk_file_path = "RK/RK_values.csv"
std_rk_file_path = "RK/std_RK_values.csv"
r_value_filtered_file_path = "RK/filtered_by_r_value.csv"
regression_results_file_path ="RK/regression_results.csv"
# Function to save dictionary as CSV
def save_dict_as_csv(dictionary, file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Key"] + list(next(iter(dictionary.values())).keys()))  # Write header
        for key, values in dictionary.items():
            writer.writerow([key] + list(values.values()))

# Save RK_values dictionary as CSV
save_dict_as_csv(RK_values, rk_file_path)

# Save std_RK_values dictionary as CSV
save_dict_as_csv(std_RK_values, std_rk_file_path)

# Save filtered_by_r_value dictionary as CSV
save_dict_as_csv(filtered_by_r_value, r_value_filtered_file_path)

save_dict_as_csv(regression_results, regression_results_file_path)
regression_results