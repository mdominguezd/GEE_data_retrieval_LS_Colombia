#!/usr/bin/env python
# coding: utf-8

# # Data retrieval of GEE data 
# The following script takes the landslides (LS) point information as an input and returns a training dataset (.geojson) with covariate data associated to each point. This data is retrieved from Google Earth Engine and consists of:
# 
# - `Slope:` Average calculated for a 1 km buffer around the LS point. (SRTM)
# - `NDVI:` Average calculated for images with low cloud probability in buffer of 1km around the LS point. (Sentinel 2)
# - `Precipitation:` Accmulated precipitation that was present in the area around the landslide one week before it happened. (GPM)
# 
# The final output of this script is the training dataset `data/Traning_LS_Dataset`


# In[2]:

import ee
import geemap
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt


def get_ndvi_filter(path_gdf = 'data/Landslide_Inventory_Colombia.geojson'):
    """
    This function takes a geojson point file and calculates the mean ndvi that has been present around every point of the file for the 7 days before a landslide occured in that area
    
    Parameters
    ----------
    path_gdf : TYPE, string
        DESCRIPTION. The default is 'data/Landslide_Inventory_Colombia.geojson'. Gets the path of a geojson file to which NDVI values will be calculated.

    Returns
    -------
    out_path
        DESCRIPTION. a string with the path of the output geojson with the values of NDVI. This geojson file will already be filtered by the points where an NDVI has been calculated

    """
    
    # Read geodataframe from path
    Landslides = gpd.read_file(path_gdf)
    
    # Import Landslide data to GEE
    LandslidesGEE = geemap.geojson_to_ee(path_gdf)
    
    # Get range of dates before landslides (one week)
    dates = Landslides['Date'].apply(lambda df : [str((df - dt.timedelta(7)))[:10], str((df + dt.timedelta(-1)))[:10]])
    
    def GetNDVIInfo(img, geo):
        # Fuction to retrieve NDVI information of the LS points from GEE using Sentinel 2.

        # Select cloudprobaBand
        msk_cloud = ee.Image(img).select('MSK_CLDPRB').divide(255).multiply(100)

        # Calculate mean cloud probability in region
        mean_cld = msk_cloud.reduceRegion(ee.Reducer.mean(), geometry = geo)
        mean_cld = mean_cld.getInfo()['MSK_CLDPRB']

        # Only calculate NDVI if cloud probability in the geometry (buffer around the LS point) is less than 7%
        if (mean_cld < 7):
            values = ee.Image(img).select(['B4','B8']).reduceRegion(ee.Reducer.mean(), geometry = geo)
            values = values.getInfo()

            # Calculate NDVI only if band values is not None
            if (values['B4'] != None)&((values['B8'] != None)):
                values = (values['B8'] - values['B4'])/(values['B8'] + values['B4'])
            else:
                values = ' '

        else:

            values = ' '
            
        # Return NDVI Values
        return values
    
    # Create empty list to append NDVI values
    NDVI_info = []
    
    # Calculate size of feature collection in GEE
    size = LandslidesGEE.size().getInfo()
    # Create a list with all of the features in the feature collection
    Ft_list = LandslidesGEE.toList(size)
    
    # Iterate over dates list
    for i in range(len(dates)):
        
        # Get the feature associated with that date and create a buffer of 1km
        ee_Ls_Feature = ee.Feature(Ft_list.get(i)).buffer(1000).geometry()
        
        # Unwrap both start and end date for the specific feature
        start_date = dates.iloc[i][0]
        end_date = dates.iloc[i][1]
        
        # Get the Image collection of sentinel for the point of interest and for the dates specified
        s2_col = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(ee_Ls_Feature).filterDate(start_date, end_date)
        
        # Calculate the size of the feature collection
        size_AOI_img = s2_col.size().getInfo()
        # Create a list of all of the images collected in the timespan
        s = s2_col.toList(size_AOI_img)
        
        # Create an empty list to append NDVI results for every image
        results_NDVI = []

        for j in range(size_AOI_img):
            
            # Calculate the NDVI value for every image in the AOI
            vals = GetNDVIInfo(s.get(j), ee_Ls_Feature)
                
            # Return a nan if value wasn't calculated
            if vals != ' ':
                results_NDVI.append(vals)
            else:
                results_NDVI.append(np.nan)
                
        # Append the mean value of NDVI in the AOI to the empty list
        NDVI_info.append(np.nanmean(results_NDVI))
        
    # Assign the NDVI_info to landslides dataset
    Landslides['S2_NDVI'][:len(NDVI_info)] = NDVI_info
    
    # Filter all values where no NDVI was calculated
    Landslides = Landslides.dropna()
    
    out_path = path_gdf.split('_')[0] + '_with_NDVI.geojson'
    
    # Export file to geojson 
    Landslides.to_file(filename = out_path, driver = 'GeoJSON')
    
    return out_path

# In[]:   
    
def get_other_covs(path_gdf = 'data/Landslide_with_NDVI.geojson'):
    """
    
    Parameters
    ----------
    path_gdf : TYPE, string
        DESCRIPTION. The default is 'data/Landslide_with_NDVI.geojson'. Gets the path of a geojson file. to which the other covariate values will be added.

    Returns
    -------
    Nothing

    """
    
    # Read geodataframe from path
    Landslides = gpd.read_file(path_gdf)
    
    # Import Landslide data to GEE
    LandslidesGEE = geemap.geojson_to_ee(path_gdf)
    
    # Get range of dates before landslides (one week)
    dates = Landslides['Date'].apply(lambda df : [str((df - dt.timedelta(7)))[:10], str((df + dt.timedelta(-1)))[:10]])
    
    ################### Retrieve Precipitation INFO ##########################
    
    def GetPrecipInfo(img, geo):
        # Fuction to retrieve precipitation information of the LS points from GEE using GPM.

        # Read image as ee.Image
        img = ee.Image(img)

        # Select precipitation band
        img = img.select('precipitationCal')

        # Get mean precipitation in the area
        values = img.reduceRegion(ee.Reducer.mean(), geometry = geo, scale = 30).getInfo()
        values = values['precipitationCal']

        if (values != None):
            return values
        else:
            return 0
        
    # Create an empty list to contain the precipitation values
    Precip = []
    
    # Calculate size of feature collection and create an iterative list with every feature
    size = LandslidesGEE.size().getInfo()
    Ft_list = LandslidesGEE.toList(size)
    
    # Iterate over the dates of the geojson file
    for i in range(len(dates)):
        
        # Get one feature
        ee_Ls_Feature = ee.Feature(Ft_list.get(i)).geometry()
        
        # Unwrap the dates (one week before the event)
        start_date = dates.iloc[i][0]
        end_date = dates.iloc[i][1]
        
        # Get the image collection with values of precipitation
        precip_col = ee.ImageCollection('NASA/GPM_L3/IMERG_V06').filterBounds(ee_Ls_Feature).filterDate(start_date, end_date)
        
        # Sum the values to get accumulated weekly precipitation
        precip_col = precip_col.sum()
        
        # Get the value of precipitation in the area
        vals = GetPrecipInfo(precip_col, ee_Ls_Feature)
        
        # Append that precipitation value to the list
        Precip.append(vals)

    # Calculate weekly accumulated precipitation and joins it to GeoDataFrame
    Landslides['Precip'] = np.array(Precip)/2
        
    ######################## Retrieve SLOPE INFO #############################
    DEM = ee.Image('USGS/SRTMGL1_003')

    # Get Colombia's Boundary
    col = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))

    # Get Slope data
    DEM = DEM.clip(col)
    Slope = ee.Terrain.slope(DEM)

    # Create buffer of every LS feature
    def apply_buffer(f):
        return(ee.Feature(f).buffer(1000))

    Ls_buffers = LandslidesGEE.map(apply_buffer)
    Ls_buffers = Ls_buffers.toList(Ls_buffers.size().getInfo())

    # Extract slope iterating over every feature
    slope = []
    for i in range(Ls_buffers.size().getInfo()):
        ls = ee.Feature(Ls_buffers.get(i)).geometry()
        slope.append(Slope.reduceRegion(ee.Reducer.mean(), geometry = ls).getInfo()['slope'])
    
    Landslides['Slope'] = slope
    
    out_path = path_gdf.split('_')[0] + '_with_covs.geojson'
    
    # Export file to geojson 
    Landslides.to_file(filename = out_path, driver = 'GeoJSON')
    
    
    return out_path

def get_full_training_set(path_LS, path_no_LS):
    
    Landslides = gpd.read_file(path_LS)
    No_LS = gpd.read_file(path_no_LS)
    
    # Select only covariate and geometry information
    Landslides['Landslide_dummy'] = 1
    No_LS['Landslide_dummy'] = 0

    train_set = pd.concat([Landslides, No_LS]).iloc[:,[0,2,3,4,5,6]]
    
    path_training_set = 'Landslide_GEE_training.geojson'
    
    train_set.to_file(filename = path_training_set, driver = 'GeoJSON')
    
    return path_training_set

    
# def get_TrainingSet(): 


# # In[4]:


#     ee.Initialize()


# # ## Import data and create functions

# # In[5]:

#     # If the script has already been ran and it has filtered the landslides where covariate information is present, only take that dataset. 
#     # If not start with the complete dataset
#     if (os.path.exists('output/Landslides_with_covs.geojson')):
#         # Import as GeoDataFrame
#         Landslides = gpd.read_file('output/Landslides_with_covs.geojson')

#         # Import Landslide data to GEE
#         LandslidesGEE = geemap.geojson_to_ee('output/Landslides_with_covs.geojson')

#     elif (os.path.exists('output/Landslides_with_NDVI.geojson')):
#         # Import as GeoDataFrame
#         Landslides = gpd.read_file('output/Landslides_with_NDVI.geojson')

#         # Import Landslide data to GEE
#         LandslidesGEE = geemap.geojson_to_ee('output/Landslides_with_NDVI.geojson')
#     else:
#         # Import as GeoDataFrame
#         Landslides = gpd.read_file('data/Landslide_Inventory_Colombia.geojson')

#         # Import Landslide data to GEE
#         LandslidesGEE = geemap.geojson_to_ee('data/Landslide_Inventory_Colombia.geojson')

#     # Get range of dates before landslides (one week)
#     dates = Landslides['Date'].apply(lambda df : [str((df - dt.timedelta(7)))[:10], str((df + dt.timedelta(-1)))[:10]])


#     # In[6]:


#     def GetNDVIInfo(img, geo):
#         # Fuction to retrieve NDVI information of the LS points from GEE using Sentinel 2.

#         # Select cloudprobaBand
#         msk_cloud = ee.Image(img).select('MSK_CLDPRB').divide(255).multiply(100)

#         # Calculate mean cloud probability in region
#         mean_cld = msk_cloud.reduceRegion(ee.Reducer.mean(), geometry = geo)
#         mean_cld = mean_cld.getInfo()['MSK_CLDPRB']

#         # Only calculate NDVI if cloud probability in the geometry (buffer around the LS point) is less than 7%
#         if (mean_cld < 7):
#             values = ee.Image(img).select(['B4','B8']).reduceRegion(ee.Reducer.mean(), geometry = geo)
#             values = values.getInfo()

#             # Calculate NDVI only if band values is not None
#             if (values['B4'] != None)&((values['B8'] != None)):
#                 values = (values['B8'] - values['B4'])/(values['B8'] + values['B4'])
#             else:
#                 values = ' '

#         else:

#             values = ' '
#         # Return NDVI Values
#         return values


#     # In[7]:


#     def GetPrecipInfo(img, geo):
#         # Fuction to retrieve precipitation information of the LS points from GEE using GPM.

#         # Read image as ee.Image
#         img = ee.Image(img)

#         # Select precipitation band
#         img = img.select('precipitationCal')

#         # Get mean precipitation in the area
#         values = img.reduceRegion(ee.Reducer.mean(), geometry = geo, scale = 30).getInfo()
#         values = values['precipitationCal']

#         if (values != None):
#             return values
#         else:
#             return 0


#     # ## NDVI data retrieval

#     # In[8]:


#     # Only perform the NDVI data retrieval once to avoid long times waiting
#     if (not os.path.exists('output/Landslides_with_NDVI.geojson')):

#         NDVI_info = []

#         for i in range(len(dates)):
#             size = LandslidesGEE.size().getInfo()

#             ee_Ls_Feature = ee.Feature(LandslidesGEE.toList(size).get(i)).buffer(1000).geometry()

#             start_date = dates.iloc[i][0]
#             end_date = dates.iloc[i][1]

#             s2_col = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(ee_Ls_Feature).filterDate(start_date, end_date)

#             size_AOI_img = s2_col.size().getInfo()
#             s = s2_col.toList(size_AOI_img)

#             results_NDVI = []

#             for j in range(size_AOI_img):

#                 vals = GetNDVIInfo(s.get(j), ee_Ls_Feature)

#                 if vals != ' ':
#                     results_NDVI.append(vals)
#                 else:
#                     results_NDVI.append(np.nan)

#             NDVI_info.append(np.nanmean(results_NDVI))
#         Landslides['S2_NDVI'] = np.ones(len(Landslides))*-9999
#         Landslides['S2_NDVI'][:len(NDVI_info)] = NDVI_info
#         Landslides = Landslides.dropna()

#         Landslides.to_file(filename = 'output/Landslides_with_NDVI.geojson', driver = 'GeoJSON')


#     # ## Slope data retrieval

#     # In[9]:


#     # Update dates list and LandslidesGEE (Filtered with data of NDVI)
#     dates = Landslides['Date'].apply(lambda df : [str((df - dt.timedelta(7)))[:10], str((df + dt.timedelta(-1)))[:10]])
#     LandslidesGEE = geemap.geojson_to_ee('output/Landslides_with_NDVI.geojson')


#     # In[10]:


#     # Only perform the Slope data retrieval once to avoid long times waiting
#     if not os.path.exists('output/Landslides_with_covs.geojson'):
#         # Retrieve SRTM DEM
#         DEM = ee.Image('USGS/SRTMGL1_003')

#         # Get Colombia's Boundary
#         col = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))

#         # Get Slope data
#         DEM = DEM.clip(col)
#         Slope = ee.Terrain.slope(DEM)

#         # Create buffer of every LS feature
#         def apply_buffer(f):
#             return(ee.Feature(f).buffer(1000))

#         Ls_buffers = LandslidesGEE.map(apply_buffer)
#         Ls_buffers = Ls_buffers.toList(Ls_buffers.size().getInfo())

#         # Extract slope iterating over every feature
#         slope = []
#         for i in range(Ls_buffers.size().getInfo()):
#             ls = ee.Feature(Ls_buffers.get(i)).geometry()
#             slope.append(Slope.reduceRegion(ee.Reducer.mean(), geometry = ls).getInfo()['slope'])

#         Landslides['Slope'] = slope


#     # ## Precipitation data retrieval

#     # In[11]:


#     # Only perform the Slope data retrieval once to avoid long times waiting
#     if not os.path.exists('output/Landslides_with_covs.geojson'):

#         Precip = []

#         for i in range(len(dates)):

#             size = LandslidesGEE.size().getInfo()

#             ee_Ls_Feature = ee.Feature(LandslidesGEE.toList(size).get(i)).geometry()

#             start_date = dates.iloc[i][0]
#             end_date = dates.iloc[i][1]

#             precip_col = ee.ImageCollection('NASA/GPM_L3/IMERG_V06').filterBounds(ee_Ls_Feature).filterDate(start_date, end_date).sum()

#             vals = GetPrecipInfo(precip_col, ee_Ls_Feature)

#             Precip.append(vals)

#         # Calculate weekly accumulated precipitation and jois it to GeoDataFrame
#         Landslides['Precip'] = np.array(Precip)/2


#     # In[12]:


#     # Export data to geojson file
#     Landslides.to_file(filename = 'output/Landslides_with_covs.geojson', driver = 'GeoJSON')


#     # In[15]:


#     Landslides.describe()


#     # ## No Landslide training dataset

#     # In[168]:


#     # Read No LS data
#     No_LS = gpd.read_file('data/No_Landside_zones.geojson')
#     No_LS_GEE = geemap.geojson_to_ee('data/No_Landside_zones.geojson')


#     # In[169]:


#     No_LS['Date'] = pd.concat([pd.DataFrame(Landslides['Date']), pd.DataFrame(Landslides['Date'])]).reset_index().iloc[:,1:]
#     dates = No_LS['Date'].apply(lambda df : [str((df - dt.timedelta(7)))[:10], str((df + dt.timedelta(-1)))[:10]])


#     # In[170]:


#     NDVI_info = []

#     for i in range(len(dates)):
#         size = No_LS_GEE.size().getInfo()

#         ee_Ls_Feature = ee.Feature(No_LS_GEE.toList(size).get(i)).buffer(1000).geometry()

#         start_date = dates.iloc[i][0]
#         end_date = dates.iloc[i][1]

#         s2_col = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(ee_Ls_Feature).filterDate(start_date, end_date)

#         size_AOI_img = s2_col.size().getInfo()
#         s = s2_col.toList(size_AOI_img)

#         results_NDVI = []

#         for j in range(size_AOI_img):

#             vals = GetNDVIInfo(s.get(j), ee_Ls_Feature)

#             if vals != ' ':
#                 results_NDVI.append(vals)
#             else:
#                 results_NDVI.append(np.nan)

#         NDVI_info.append(np.nanmean(results_NDVI))


#     # In[171]:


#     No_LS['S2_NDVI'] = NDVI_info
#     No_LS = No_LS.dropna()

#     No_LS.to_file(filename = 'output/No_Landslides_with_NDVI.geojson', driver = 'GeoJSON')


#     # In[172]:


#     dates = No_LS['Date'].apply(lambda df : [str((df - dt.timedelta(7)))[:10], str((df + dt.timedelta(-1)))[:10]])
#     No_LS_GEE = geemap.geojson_to_ee('output/No_Landslides_with_NDVI.geojson')


#     # In[173]:


#     Precip = []

#     for i in range(len(dates)):

#         size = No_LS_GEE.size().getInfo()

#         ee_Ls_Feature = ee.Feature(No_LS_GEE.toList(size).get(i)).geometry()

#         start_date = dates.iloc[i][0]
#         end_date = dates.iloc[i][1]

#         precip_col = ee.ImageCollection('NASA/GPM_L3/IMERG_V06').filterBounds(ee_Ls_Feature).filterDate(start_date, end_date).sum()

#         vals = GetPrecipInfo(precip_col, ee_Ls_Feature)

#         Precip.append(vals)

#     # Calculate weekly accumulated precipitation and jois it to GeoDataFrame
#     No_LS['Precip'] = np.array(Precip)/2


#     # In[174]:


#     # Retrieve SRTM DEM
#     DEM = ee.Image('USGS/SRTMGL1_003')

#     # Get Colombia's Boundary
#     col = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))

#     # Get Slope data
#     DEM = DEM.clip(col)
#     Slope = ee.Terrain.slope(DEM)

#     # Create buffer of every LS feature
#     def apply_buffer(f):
#         return(ee.Feature(f).buffer(1000))

#     buffers = No_LS_GEE.map(apply_buffer)
#     buffers = buffers.toList(buffers.size().getInfo())

#     # Extract slope iterating over every feature
#     slope = []
#     for i in range(buffers.size().getInfo()):
#         ls = ee.Feature(buffers.get(i)).geometry()
#         val = Slope.reduceRegion(ee.Reducer.mean(), geometry = ls).getInfo()['slope']
#         slope.append(val)

#     No_LS['Slope'] = slope


#     # In[175]:


#     No_LS.to_file(filename = 'output/No_Landslides_with_covs.geojson',
#                   driver = 'GeoJSON')


#     # ## Merging and export training Dataset

#     # In[176]:


#     # Select only covariate and geometry information
#     Landslides['Landslide_dummy'] = 1
#     No_LS['Landslide_dummy'] = 0

#     train_set = pd.concat([Landslides, No_LS]).iloc[:,[0,2,3,4,5,6]]


#     # In[177]:


#     train_set.to_file('data/Training_LS_Dataset.geojson', 
#                       driver = 'GeoJSON')


#     # In[178]:


#     train_set.groupby('Landslide_dummy').median()


#     # In[182]:


#     train_set


#     # ## Visualize the data

#     # In[238]:


#     # Get Colombia's Boundary for plot
#     world_filepath = gpd.datasets.get_path('naturalearth_lowres')
#     world = gpd.read_file(world_filepath)
#     countries = world.loc[(world['name'] == 'Colombia')|
#                           (world['name'] == 'Venezuela')|
#                           (world['name'] == 'Panama')|
#                           (world['name'] == 'Ecuador')|
#                           (world['name'] == 'Brazil')|
#                           (world['name'] == 'Peru')
#                          ]

#     # Create plot 
#     fig, ax = plt.subplots(1,1,  dpi = 250) #Create figure
#     fig.set_size_inches(8,8) #Change figure size 
#     base = countries.plot(column = 'name',
#                           edgecolor = 'k',
#                           facecolor = 'w',
#                           cmap = 'Pastel2',
#                           ax = ax,
#                           legend = True,
#                          ) # Use Colombia's Boundary as basemap

#     plt.xlim(-82.5,-65)
#     plt.ylim(-5,15)

#     cols = {0 : 'red', 1 : 'blue'}

#     train_set.plot(kind = 'geo',
#                    ax = base,
#                    legend = True,
#                    column = train_set['Landslide_dummy'].apply(lambda l : 'Landslide' if l == 1 else 'No Landslide'),
#                    categorical = True,
#                    cmap = 'Set1',
#                    legend_kwds={'loc': 'center left', 'bbox_to_anchor':(1,0.5),'fmt': "{:.0f}"},
#                    edgecolor = 'k'
#                   ) # Plot 
    
#     fig.savefig('figures/Complete_TrainingDataset.png')





