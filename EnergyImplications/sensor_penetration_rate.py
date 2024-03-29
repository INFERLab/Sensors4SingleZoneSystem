#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:20:13 2024

@author: ozanbaris
This script provides plots for the penetration rates of remote sensors and DR participation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

metadata_df = pd.read_csv('meta_data.csv')
#%%

def summarize_column(column_name, dataframe):
    # Drop rows where the specified column is NaN
    cleaned_df = dataframe.dropna(subset=[column_name])
    
    # Calculate counts for each unique value in the specified column
    value_counts = cleaned_df[column_name].value_counts().sort_index()
    
    # Calculate percentages
    value_percentages = cleaned_df[column_name].value_counts(normalize=True).sort_index() * 100
    
    # Calculate cumulative counts and percentages in the correct order for "more than or equal to"
    cumulative_counts = value_counts.sort_index(ascending=False).cumsum().sort_index(ascending=True)
    cumulative_percentages = value_percentages.sort_index(ascending=False).cumsum().sort_index(ascending=True)
    
    # Create a DataFrame to hold the calculated values
    summary_table = pd.DataFrame({
        'Count': value_counts,
        'Percentage': value_percentages,
        'Cumulative Count': cumulative_counts,
        'Cumulative Percentage': cumulative_percentages
    })
    
    print("Summary Table for column:", column_name)
    print(summary_table)


def plot_distribution(column_name, dataframe, reindex=False):
    """
    Plots the distribution and cumulative percentage of a specified column in a dataframe.
    
    Parameters:
    - column_name: The name of the column to analyze.
    - dataframe: The DataFrame containing the column.
    - reindex: A boolean flag to indicate whether to reindex the column values to include a '5+' category.
    """
    if reindex:
        # Adjusting the dataframe copy to avoid SettingWithCopyWarning
        df_temp = dataframe.copy()
        df_temp[column_name] = df_temp[column_name].apply(lambda x: '5+' if x >= 5 else str(x))
        value_counts = df_temp[column_name].value_counts().reindex(['0', '1', '2', '3', '4', '5+'], fill_value=0)
    else:
        value_counts = dataframe[column_name].value_counts().sort_index()
    
    # Total count for normalization
    total_count = value_counts.sum()

    # Calculate percentages
    value_percentages = value_counts / total_count * 100

    # Calculate cumulative counts in reverse order
    cumulative_counts = value_counts.iloc[::-1].cumsum()[::-1]

    # Calculate cumulative percentages based on cumulative counts
    cumulative_percentages = cumulative_counts / total_count * 100

    # Plotting
    fig, ax1 = plt.subplots(figsize=(15, 10))

    # Bar chart for counts
    bars = ax1.bar(value_counts.index, value_counts, color='#00008B', label='Count')

    # Annotate bars with percentage values, adjusted for various index types
    for bar, percentage in zip(bars, value_percentages):
        height = bar.get_height()
        ax1.annotate(f'{percentage:.1f}%', 
                     xy=(bar.get_x() + bar.get_width() / 2, height / 2), 
                     ha='center', va='center', 
                     color='white', weight='bold', fontsize=25)


    # Line plot for cumulative percentages
    ax2 = ax1.twinx()
    ax2.plot(value_counts.index, cumulative_percentages, color='#FA8072', marker='o', label='Cumulative Percentage', lw=2)
    ax2.set_ylabel('Cumulative Percentage (%)', color='#8B0000')
    ax2.tick_params(axis='y', colors='#8B0000')

    # Corrected annotation for cumulative percentages
    for label, txt in zip(value_counts.index, cumulative_percentages):
        x_pos = label  # Use label directly for x position
        y_pos = txt
        ax2.annotate(f'{txt:.1f}%', 
                     xy=(x_pos, y_pos), 
                     textcoords="offset points", xytext=(0,10), 
                     ha='center', va='bottom', 
                     color='#FA8072', weight='bold', fontsize=25)
    # Legend
    legend_handles = [Patch(facecolor='#00008B', edgecolor='#00008B', label='Count'),
                      Line2D([0], [0], color='#FA8072', marker='o', label='Cumulative Percentage')]
    ax1.legend(handles=legend_handles, loc='upper right')

    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of {column_name}')
    ax1.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
    plt.savefig(f'{column_name}_dist.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
plt.rcParams['font.size'] = 26



#%%
summarize_column('Number of Remote Sensors', metadata_df)
plot_distribution('Number of Remote Sensors', metadata_df, reindex=True)
#%%
summarize_column('eco+ slider level', metadata_df)
plot_distribution('eco+ slider level', metadata_df, reindex=False)
#%%
# Calculate the number of non-NaN values in the 'eco+ slider level' column
non_nan_count = metadata_df['eco+ slider level'].notnull().sum()

# Calculate the total number of rows in the DataFrame to determine the denominator
total_rows = len(metadata_df)

# Calculate the percentage of non-NaN values in the 'eco+ slider level' column
non_nan_percentage = (non_nan_count / total_rows) * 100

print(f"Number of non-NaN values in 'eco+ slider level': {non_nan_count}")
print(f"Percentage of non-NaN values: {non_nan_percentage:.2f}%")

#%%

# Group the 'Number of Remote Sensors' into '0', '1', '2', '3', '4', '5+'
metadata_df['Grouped Sensors'] = metadata_df['Number of Remote Sensors'].apply(lambda x: '5+' if x >= 5 else str(x))

# Count occurrences of eco+ slider level within each Grouped Sensors category
distribution_df = metadata_df.groupby(['Grouped Sensors', 'eco+ slider level']).size().unstack(fill_value=0)

# Calculate the percentage of each eco+ slider level within the groups
distribution_percentage = distribution_df.div(distribution_df.sum(axis=1), axis=0) * 100


# Custom colors for each eco+ slider level - adjust these as needed
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Ensure there are enough colors for the eco+ slider levels
if len(custom_colors) < len(distribution_df.columns):
    print("Warning: Not enough custom colors for each eco+ slider level. Please add more colors.")
else:
    custom_colors = custom_colors[:len(distribution_df.columns)]


# Plot adjustments
ax = distribution_df.plot(kind='bar', stacked=True, figsize=(16, 14), color=custom_colors)
plt.xlabel('Number of Remote Sensors')
plt.ylabel('Count of Houses')
plt.xticks(rotation=0)  # Make x-tick labels horizontal
ax.legend(title='Eco+ Slider Level', loc='upper right')

# Initialize a zero array to keep track of the cumulative heights for annotations
y_offset = np.zeros(len(distribution_df))

# Correctly annotate each part of the stack
for sensor_group_index, (index_value, row) in enumerate(distribution_df.iterrows()):
    for eco_slider_level, value in enumerate(row):
        count = value
        total = row.sum()
        percentage = (count / total) * 100 if total > 0 else 0

        # Only proceed if there's a count to display
        if count > 0:
            # Calculate the height at which to place the annotation
            height = y_offset[sensor_group_index] + (count / 2)

            # Update the y_offset for the next bar part
            y_offset[sensor_group_index] += count

            # Placing text annotation
            ax.text(x=sensor_group_index, y=height, s=f'{count} ({percentage:.1f}%)',
                    ha='center', va='center', fontsize=16, fontweight='bold', color='black', rotation=0)

plt.tight_layout()
plt.show()
#%%

#  Filter for non-NaN eco+ slider values
eco_plus_enrolled = metadata_df.dropna(subset=['eco+ slider level'])
eco_plus_enrolled['Number of Remote Sensors'] = pd.to_numeric(eco_plus_enrolled['Number of Remote Sensors'], errors='coerce')

#  Count houses with 1 or more sensors
houses_with_1_or_more_sensors = eco_plus_enrolled[eco_plus_enrolled['Number of Remote Sensors'] >= 1].shape[0]

#  Count houses with 2 or more sensors
houses_with_2_or_more_sensors = eco_plus_enrolled[eco_plus_enrolled['Number of Remote Sensors'] >= 2].shape[0]

# Total number of houses enrolled in eco+
total_eco_plus_houses = eco_plus_enrolled.shape[0]

# Step 4: Calculate cumulative percentages
percentage_with_1_or_more_sensors = (houses_with_1_or_more_sensors / total_eco_plus_houses) * 100
percentage_with_2_or_more_sensors = (houses_with_2_or_more_sensors / total_eco_plus_houses) * 100

percentage_with_1_or_more_sensors, percentage_with_2_or_more_sensors


#%%
import pandas as pd

# Assuming metadata_df is your DataFrame
numerical_columns = ['Floor Area [ft2]', 'Number of Floors', 'Age of Home [years]',
                     'Number of Occupants', 'installedCoolStages', 'installedHeatStages',
                     'Number of Remote Sensors']

# Print min and max values for each numerical column
for column in numerical_columns:
    min_value = metadata_df[column].min()
    max_value = metadata_df[column].max()
    print(f"{column}: Min = {min_value}, Max = {max_value}")
#%%

numerical_columns = ['Number of Floors', 'Age of Home [years]',
                     'Number of Occupants', 'installedCoolStages', 'installedHeatStages',
                     'Number of Remote Sensors']

# Removing rows where 'Floor Area [ft2]' is 0
metadata_df = metadata_df[metadata_df['Floor Area [ft2]'] != 0]

for column in numerical_columns:
    Q1 = metadata_df[column].quantile(0.25)
    Q3 = metadata_df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    metadata_df = metadata_df[(metadata_df[column] >= lower_bound) & (metadata_df[column] <= upper_bound)]

#  Assign False to NaN values in 'eco+ enrolled'
metadata_df['eco+ enrolled'] = metadata_df['eco+ enrolled'].fillna(False)

#  Remove rows with NaN values except in 'eco+ slider level' and keeping rows with NaN in 'Style'
columns_except_slider_and_style = metadata_df.columns.difference(['eco+ slider level', 'Style'])
metadata_df = metadata_df.dropna(subset=columns_except_slider_and_style)

# Display the cleaned dataset
print(metadata_df.head())


#%%
import pandas as pd

# Function to categorize 'Number of Remote Sensors'
def categorize_sensors(x):
    if x >= 5:
        return '5+'
    else:
        return str(x)

# Function to categorize 'Number of Floors'
def categorize_floors(x):
    if x >= 4:
        return '4+'
    else:
        return str(x)

# Applying categorizations
metadata_df['Sensor Category'] = metadata_df['Number of Remote Sensors'].apply(categorize_sensors)
metadata_df['Floors Category'] = metadata_df['Number of Floors'].apply(categorize_floors)
metadata_df['Number of Occupants'] = metadata_df['Number of Occupants'].apply(categorize_sensors)
# Binning 'Floor Area [ft2]'
bins = [100, 1000, 2000, 3000, 4000, float('inf')]
labels = ['100-1000', '1000-2000', '2000-3000', '3000-4000', '4000+']
metadata_df['Floor Area Category'] = pd.cut(metadata_df['Floor Area [ft2]'], bins=bins, labels=labels, right=False)

# Display the modified DataFrame
print(metadata_df)


#%%
import pandas as pd
import matplotlib.pyplot as plt

def plot_category_charts(metadata_df, cols, exclude_zero_occupants=False):
    # Titles for the plots
    plot_titles = {
        'Number of Occupants': 'Number of Occupants',
        'Floors Category': 'Number of Floors',
        'Floor Area Category': 'Floor Area'
    }
    
    fig, axs = plt.subplots(2, len(cols), figsize=(10 * len(cols), 16), sharey=False)
    
    for i, occupant_col in enumerate(cols):
        # Exclude rows where 'Number of Occupants' is 0 if the flag is True
        if exclude_zero_occupants and occupant_col == 'Number of Occupants':
            df_filtered = metadata_df[metadata_df[occupant_col] != '0']
        else:
            df_filtered = metadata_df.copy()
        
        # Prepare data for the first stacked bar chart (Proportions)
        proportion_data = df_filtered.groupby('Sensor Category')[occupant_col].value_counts(normalize=True).unstack(fill_value=0)
        
        # Prepare data for the second stacked bar chart (Frequencies)
        frequency_data = df_filtered.groupby('Sensor Category')[occupant_col].value_counts().unstack(fill_value=0)
        
        # Plot the second stacked bar chart (Frequencies)
        bars = frequency_data.plot(kind='bar', stacked=True, ax=axs[0, i])
        axs[0, i].set_ylabel('Frequency')
        axs[0, i].set_xticklabels([])  # Remove x-axis labels
        axs[0, i].set_xlabel('')  # Remove x-axis title
        # Adjusting the legend to use 2 columns
        axs[0, i].legend(title=plot_titles[occupant_col], ncol=2)
        
        # Plot the first stacked bar chart (Proportions)
        bars = proportion_data.plot(kind='bar', stacked=True, ax=axs[1, i], legend=False)
        axs[1, i].set_xlabel('Number of Remote Sensors')
        axs[1, i].set_ylabel('Proportion')
        axs[1, i].tick_params(axis='x', rotation=0)  # Keep x-axis labels straight
        axs[1, i].set_ylim(ymax=1)
        # Add annotations for proportions with customized font
        for bar in bars.containers:
            axs[1, i].bar_label(bar, fmt='%.2f', label_type='center', color='white', fontsize=9, weight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{occupant_col.replace(' ', '_').lower()}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage with the flag
cols = ['Number of Occupants', 'Floors Category', 'Floor Area Category']
plot_category_charts(metadata_df, cols, exclude_zero_occupants=True)

