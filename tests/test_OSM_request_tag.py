import osmnx as ox
import geopandas as gpd
import pandas as pd
from osmnx._errors import InsufficientResponseError as NoResponse
from constants import OSM_TAGS
from shapely.ops import transform
from constants import GLOBAL_CRS, GLOBAL_METRIC_CRS, GLOBAL_EPSG
from tqdm import tqdm
import sys

class GeoDataGetter:
    def get_features_from_id(
                    osm_id:int
                   ,tags:dict
                   ,osm_type = "R"
                   ,selected_columns = ['tag','element_type', 'osmid', 'name', 'geometry', 'centroid']
                   ) -> gpd.GeoDataFrame:
          
        place = ox.project_gdf(ox.geocode_to_gdf(osm_type+str(osm_id), by_osmid=True))
        place_name = place.name.iloc[0]
        gdf_list = []
        for category, category_tags in tags.items():
            for tag in tqdm(category_tags, desc=f'Processing category {category}'):
                try:
                    gdf = ox.features_from_place(place_name, tags={category: tag})
                    gdf.geometry.dropna(inplace=True)
                    gdf['tag'] = category
                    gdf['centroid'] = gdf['geometry']
                    
                    tmpgdf = ox.projection.project_gdf(gdf, to_crs=GLOBAL_METRIC_CRS, to_latlong=False)
                    tmpgdf['centroid'] = tmpgdf['geometry'].centroid
                    tmpgdf = tmpgdf.to_crs(GLOBAL_CRS)
                    gdf['centroid'] = tmpgdf['centroid']
                    
                    tmpgdf = None
                    gdf_list.append(gdf)
                    
                except (NoResponse, AttributeError):
                    print(f'\nFailed to export {category}-{tag}\nException Info:\n{chr(10).join([str(line) for line in sys.exc_info()])}')
                    pass
                
        if len(gdf_list) > 0:
            merged_gdf = pd.concat(gdf_list).reset_index().loc[:,selected_columns]
        else:
            merged_gdf = pd.DataFrame(columns=selected_columns)
            
        
        return merged_gdf

