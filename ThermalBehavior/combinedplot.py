#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:35:52 2023

@author: ozanbaris
#This code will Plot a combined plot of distribution of deviations from heating and cooling set point. 
#You should first run cooling_comfort.py and heating_comfort.py to get house_dictionaries and new_house_dictionaries for this plot to work.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def plot_temperature_deviations(house_dicts, season_name, axs, season_type='cooling'):
    # Colors for each category
    colors = [(0,0,1,0.5), (1,0,0,0.5), (0,1,0,0.5), (1,0,1,0.5), (0,1,1,0.5)]
    position = 0
    positions = []
    separator_positions = []

    # Setpoint variable based on season type
    setpoint_var = 'Indoor_CoolSetpoint' if season_type == 'cooling' else 'Indoor_HeatSetpoint'

    # House counts based on season type
    house_counts = [387, 85, 308, 42, 65] if season_type == 'cooling' else [350, 93, 285, 42, 75]

    for i, houses in enumerate(house_dicts):
        deviations_houses = {}
        for house_id, data in houses.items():
            deviations = {}
            deviations['Thermostat_Temperature'] = data['Thermostat_Temperature'] - data[setpoint_var]
            for sensor in range(1, i+2):
                sensor_column = f'RemoteSensor{sensor}_Temperature'
                if sensor_column in data.columns:
                    deviations[sensor_column] = data[sensor_column] - data[setpoint_var]

            deviations_df = pd.DataFrame(deviations)
            deviations_houses[house_id] = deviations_df

        all_deviations = pd.concat(deviations_houses.values())
        old_names = ['Thermostat_Temperature'] + [f'RemoteSensor{i}_Temperature' for i in range(1, i+2)]

        bplot = axs.boxplot([all_deviations[col].dropna() for col in old_names],
                            positions=[j+1+position for j in range(len(old_names))],
                            widths=0.6, notch=True, patch_artist=True,
                            boxprops=dict(facecolor=colors[i], linewidth=2.5),
                            medianprops=dict(color='black', linewidth=2.5),
                            whiskerprops=dict(linewidth=2.5),
                            capprops=dict(linewidth=2),
                            whis=1.5, showfliers=False)

        for j in range(len(old_names)):
            positions.append(j+1+position)
        separator_positions.append(positions[-1] + 0.5)
        position += len(old_names) + 0.1

    axs.axhline(0, color='gray', linestyle='--', linewidth=2)

    x_ticks_labels = []
    for i in range(5):
        x_ticks_labels += ['T'] + [f'{j}' for j in range(1, i+2)]
    axs.set_xticks(positions)
    axs.set_xticklabels(x_ticks_labels, fontsize=27)
    axs.tick_params(axis='y', labelsize=27)
    axs.tick_params(axis='x', which='both', width=2, length=6)  # Thicker x-axis ticks

    for pos in separator_positions[:-1]:
        axs.axvline(pos, color='gray', linestyle='-.', linewidth=1.5)


    
    # Set legend only for heating season
    if season_type == 'heating':
        patches = [mpatches.Patch(color=colors[i], label=f'{house_counts[i]}', alpha=0.4) for i in range(5)]
        legend = axs.legend(handles=patches, fontsize=24, title='Number of Houses', title_fontsize=24, ncol=3)
    else: 
        patches = [mpatches.Patch(color=colors[i], label=f'{house_counts[i]}', alpha=0.4) for i in range(5)]
        legend = axs.legend(handles=patches, fontsize=24, title='Number of Houses', title_fontsize=24, ncol=2)
        legend.set_bbox_to_anchor((0.45, 0.35))
    axs.text(0.01, 0.97, season_name, transform=axs.transAxes, fontsize=27, verticalalignment='top')

    fig, (axs1, axs2) = plt.subplots(nrows=2, figsize=(16, 16), sharex=True)

    # Set individual y-axis labels
    axs1.set_ylabel('Deviation from Cooling Setpoint (°F)', fontsize=27)
    axs2.set_ylabel('Deviation from Heating Setpoint (°F)', fontsize=27)

    # Remove x-axis label from top subplot
    axs1.set_xlabel("")
    axs2.set_xlabel('Sensor', fontsize=27)
    plt.tight_layout()
    plt.savefig("sensors.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


house_dictionaries = [one_houses, two_houses, three_houses, four_houses, five_houses]
plot_temperature_deviations(house_dictionaries, 'Cooling Season', axs1, season_type='cooling')

new_house_dictionaries = [heat_one_houses, heat_two_houses, heat_three_houses, heat_four_houses, heat_five_houses]
plot_temperature_deviations(new_house_dictionaries, 'Heating Season', axs2, season_type='heating')
