#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import built-in modules
import sys
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt 


# In[2]:


# Setup working environment
if 'data' not in os.listdir():
    os.mkdir('data')
if 'output' not in os.listdir():
    os.mkdir('output')
if 'figures' not in os.listdir():
    os.mkdir('figures')


# In[3]:

def get_LS_data():
    # Read the dataset downloded
    df = pd.read_excel('RAW_data/Landslides_DataBase.xlsx')

    # Merge all excel sheets into one
    for i in np.arange(1,37,1):
        df = pd.concat([df, pd.read_excel('RAW_data/Landslides_DataBase.xlsx', sheet_name = i)])

    df.sort_values('Total de daños')


    # In[4]:


    # Set date as date type
    df['Date'] = df['Fecha evento'].apply(lambda df : pd.to_datetime(df, format = '%d/%m/%Y'))
    # Translate type of mass movement (Landslide == Deslizamiento) else we are not interested
    df['Type'] = df.apply(lambda df : 'Landslide' if df['Tipo movimiento del primer'] == 'Deslizamiento' else np.nan, axis = 1)


    # In[5]:


    # Transform dataframe to geo-dataframe
    landslides_geo = gpd.GeoDataFrame(df[['Date','Type']], geometry=gpd.points_from_xy(df['Longitud (°)'], df['Latitud (°)']))

    # Export geodataframe as .GEOJSON
    landslides_geo.to_file('data/Landslide_Inventory_Colombia.geojson', driver = 'GeoJSON')


    # In[6]:


    # Get Colombia's Boundary for plot
    world_filepath = gpd.datasets.get_path('naturalearth_lowres')
    world = gpd.read_file(world_filepath)
    colombia = world.loc[world['name'] == 'Colombia']

    # Create plot 
    fig, ax = plt.subplots(1,1) #Create figure
    fig.set_size_inches(8,8) #Change figure size 
    base = colombia.plot(column = 'name',
                         edgecolor = 'k',
                         facecolor = 'w',
                         cmap = 'Reds',
                         ax = ax,
                         legend = True
                        ) # Use Colombia's Boundary as basemap
    landslides_geo.plot(kind = 'geo',
                        column = 'Type',
                        ax = base,
                        legend = True,
                        edgecolor = 'k'
                       ) # Plot 

    # Set figure characteristics
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title('Landslides in Colombia [2017 - 2022]')

    # Save figure as png file
    fig.savefig('figures/LandslideMapColombia_2017_2022.png')





