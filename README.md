`TITLE:` Near-real time landslide predictions in Colombia. Results of a case study in the Cauca, region.
<br>
`TEAMS:` Piquant rainbow lobster & Geoscripters 2023
<br>
    

    
# WHY? - *The aim or question of the project*

The aim of our project is to use landslide point data of Colombia and relate various factors which may potentially contribute to increased landslide risk, such as: slope, precipitation, vegetation before the event and fault proximity. We would use this relationship to identify areas at risk of landslides in the Cauca region of Colombia in near-real time. 

# WHAT? - *The datasets & metadata (author, date, extent, resolution)*

| Datasets                                                      | Purpose         | Link                                                                                                                                                         | Author                     | Date        | Extent   | Resolution | Size   |
|---------------------------------------------------------------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|-------------|----------|------------|--------|
| Landslide event point with coordinates and date of occurrence | Training points | https://simma.sgc.gov.co/#/                                                                                                                                  | Colombian Geologic Service | 2017 - 2022 | Colombia |            | 4.5MB  |
| No Landslide points                                           | Training points | Randomly selected in areas without landslides                                                                                                                |                            | 2017 - 2022 | Colombia |            |        |
| Slope                                                         | Predictor       | https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003                                                                                 | SRTM - NASA                | 2000        | Colombia | 30 m       | GEE    |
| Near real time precipitation                                  | Predictor       | https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06                                                                            | NASA GES DISC              | 2000-2023   | Colombia | 10km       | GEE    |
| NDVI                                                          | Predictor       | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR                                                                                 | ESA- Sentinel 2            | 2016 - 2023 | Colombia | 10 m       | GEE    |
| Faults proximity                                              | Predictor       | https://datos-sgcolombiano.opendata.arcgis.com/datasets/e03339c845d24e7baceb6d67397a23b3_0.geojson?outSR=%7B%22latestWkid%22%3A3857%2C%22wkid%22%3A102100%7D | Colombian geologic service | 2020        | Colombia |            | 4.5 MB |

# HOW? - *Methods and potential results*

## Step by step

1. Retrieve landslide information. Where have they happened and when? (`DONE` for all Colombia from 2017 to 2022) [Excel file to geojson] 
1. Create synthetic data of no landslides for areas where there have not been landslides in our time of study. (`DONE` - R code in repository) 
1. Retrieve GEE information for both landslide and no landslide datasets (`DONE`) 
    1. This is the process that takes the longest because it calculates NDVI, slope and accumulated precipitation of an area of interest around the point provided.  
    1. The output here is a training dataset with 4 columns: 
        1. `Landslide_dummy:` 0 or 1 if the event happened or not. 
        1. `NDVI:` A value of NDVI from the week before the event occurred or not. 
        1. `Slope:` A mean value of slope in the point of interest (POI).
        1. `Precipitation:` An accumulated value of precipitation for the week before the event. 
1. Retrieve additional predictors, for example distance to faults. (R code to get faults in repo, but missing distance info and assign it to training dataset) 
1. Train **logistic regression model** with training dataset (y = Landslide_dummy,  X = all the predictors) 
    1. Use of **sci-kit learn** package in Python to fit logistic regression. 
1. Calculate odds ratio and present them as a result. 
1. Apply the model for an area of interest. For this we would need: 
    1. One cloud free (as free as we can find) sentinel 2 image to calculate the NDVI. 
    1. Slope SRTM data 
    1. Precipitation values (data from GPM) (Weekly accumulated) 
    1. Fault proximity raster 

1. Create a dynamic map of landslide prone areas in the area at the moment. 
1. If we have time create an app to allow the user to decide an area where to vsualize the landslide prone zones.
    
## Potential results
- Quantification of the effect of a change in predictors (increase or decrease in slope, precipitation, vegetation) in the odds of having a landslide. 
- Near-real time probability map of landslide hazard in study area. 
"# GEE_data_retrieval_LS_Colombia" 
