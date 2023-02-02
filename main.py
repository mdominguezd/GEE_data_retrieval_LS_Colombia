
import ee

# import Python
import Python.PreProcessing_LandslideData as Pre_LS
import Python.GEE_DataRetrieval as Pre_GEE

# ee.Authenticate()
ee.Initialize()

# Pre-process LS raw data
LS_path = Pre_LS.get_LS_data()

# Get data filtered by NDVI values
LS_filtered = Pre_GEE.get_ndvi_filter(LS_path)

# Get data for all Landslides filtered by NDVI
LS_complete = Pre_GEE.get_other_covs(LS_filtered)

# Get data filtered by NDVI for No Landslides dataset
No_LS_path = 'data/No_Landslide_zones.geojson'
No_LS_filtered = Pre_GEE.get_ndvi_filter(No_LS_path)

# Get covariate data for the no Landslide points filtered by NDVI
No_LS_complete = Pre_GEE.get_other_covs(No_LS_filtered)

# Get and download the complete training dataset
Train = Pre_GEE.get_full_training_set(LS_complete, No_LS_complete)

