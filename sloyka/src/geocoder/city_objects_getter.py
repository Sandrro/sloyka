from typing import List

import pandas as pd
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
from loguru import logger
from natasha import MorphVocab
from sloyka.src.utils.constants import NUM_CITY_OBJ
from sloyka.src.geocoder.address_extractor_titles import AddrNEWExtractor
import numpy as np

class OtherGeoObjects:
    @staticmethod
    def get_OSM_green_obj(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about green_obj.
        """
        tags = {'leisure': ['park', 'garden', 'recreation_ground']}
        green_obj = ox.geometries_from_place(osm_city_name, tags)
        osm_green_obj_df = pd.DataFrame(green_obj)
        osm_green_obj_df = osm_green_obj_df.dropna(subset=['name'])
        osm_green_obj_df = osm_green_obj_df[['name', 'geometry', 'leisure']]
        return osm_green_obj_df
    
    @staticmethod
    def get_OSM_num_obj(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about amenity.
        """
        tags = {'amenity': ['hospital', 'clinic', 'school', 'kindergarten']}
        osm_num_obj = ox.geometries_from_place(osm_city_name, tags)
        osm_num_obj_df = pd.DataFrame(osm_num_obj)
        osm_num_obj_df = osm_num_obj_df.dropna(subset=['name'])
        osm_num_obj_df = osm_num_obj_df[['name', 'geometry', 'amenity']]
        return osm_num_obj_df
    
    @staticmethod
    def get_OSM_cemetery(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about cemetery.
        """
        tags = {'landuse': ['cemetery']}
        osm_cemetery = ox.geometries_from_place(osm_city_name, tags)
        osm_cemetery_df = pd.DataFrame(osm_cemetery)
        osm_cemetery_df = osm_cemetery_df.dropna(subset=['name'])
        osm_cemetery_df = osm_cemetery_df[['name', 'geometry', 'landuse']]
        return osm_cemetery_df
    
    @staticmethod
    def get_OSM_natural(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about natural obj.
        """
        tags = {'natural': ['beach', 'water']}
        osm_natural = ox.geometries_from_place(osm_city_name, tags)
        osm_natural_df = pd.DataFrame(osm_natural)
        osm_natural_df = osm_natural_df.dropna(subset=['name'])
        osm_natural_df = osm_natural_df[['name', 'geometry', 'natural']]
        return osm_natural_df
    
    @staticmethod
    def get_OSM_railway(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about railway obj.
        """
        tags = {'railway': ['station', 'subway']}
        osm_railway = ox.geometries_from_place(osm_city_name, tags)
        osm_railway_df = pd.DataFrame(osm_railway)
        osm_railway_df = osm_railway_df.dropna(subset=['name'])
        osm_railway_df = osm_railway_df[['name', 'geometry', 'railway']]
        return osm_railway_df
    
    @staticmethod
    def get_OSM_tourism(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about tourism obj.
        """
        tags = {'tourism': ['attraction', 'museum']}
        osm_tourism = ox.geometries_from_place(osm_city_name, tags)
        osm_tourism_df = pd.DataFrame(osm_tourism)
        osm_tourism_df = osm_tourism_df.dropna(subset=['name'])
        osm_tourism_df = osm_tourism_df[['name', 'geometry', 'tourism']]
        return osm_tourism_df
    
    @staticmethod
    def get_OSM_historic(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about historical obj.
        """
        tags = {'historic': ['monument', 'memorial']}
        osm_historic = ox.geometries_from_place(osm_city_name, tags)
        osm_historic_df = pd.DataFrame(osm_historic)
        osm_historic_df = osm_historic_df.dropna(subset=['name'])
        osm_historic_df = osm_historic_df[['name', 'geometry', 'historic']]
        return osm_historic_df

    @staticmethod    
    def get_OSM_square(osm_city_name) -> pd.DataFrame:
        """
        This function sets spatial data from OSM about square obj.
        """
        tags = {'place': ['square']}
        osm_square = ox.geometries_from_place(osm_city_name, tags)
        osm_square_df = pd.DataFrame(osm_square)
        osm_square_df = osm_square_df.dropna(subset=['name'])
        osm_square_df = osm_square_df[['name', 'geometry', 'place']]
        return osm_square_df

    @staticmethod
    def calculate_centroid(geometry) -> pd.DataFrame:
        """
        This function counts the centroid for polygons.
        """
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry.centroid
        elif isinstance(geometry, Point):
            return geometry
        else:
            return None
    
    def get_and_process_osm_data(osm_city_name, get_data_function) -> pd.DataFrame:
        """
        This function allows you to build an OSM array for different urban objects.
        """
        df = get_data_function(osm_city_name)
        df['geometry'] = df['geometry'].apply(OtherGeoObjects.calculate_centroid)
        df.rename(columns={df.columns[2]: 'geo_obj_tag'}, inplace=True)
        return df
        
    def run_OSM_dfs(osm_city_name) -> pd.DataFrame:
        """
        This function collects dataframes with OSM spatial data, finds centroids and combines files into one.
        """
        logger.info('run_OSM_dfs started')

        osm_functions = [
            OtherGeoObjects.get_OSM_green_obj,
            OtherGeoObjects.get_OSM_num_obj,
            OtherGeoObjects.get_OSM_cemetery,
            OtherGeoObjects.get_OSM_natural,
            OtherGeoObjects.get_OSM_railway,
            OtherGeoObjects.get_OSM_tourism,
            OtherGeoObjects.get_OSM_historic
        ]

        osm_dfs = [OtherGeoObjects.get_and_process_osm_data(osm_city_name, func) for func in osm_functions]
        osm_combined_df = pd.concat(osm_dfs, axis=0)

        return osm_combined_df
        
    @staticmethod
    def extract_geo_obj(text) -> List[str]:
        """
        The function extracts location entities from the text, using the Natasha library.
        """
        morph = MorphVocab()
        extractor = AddrNEWExtractor(morph)

        other_geo_obj = []

        matches = extractor(text)
        for match in matches:
            part = match.fact
            if part.value and part.type:
                combined_phrase = f"{part.value} {part.type}"
                other_geo_obj.append(combined_phrase)
            elif part.value:
                other_geo_obj.append(part.value)
            elif part.type:
                other_geo_obj.append(part.type)

        return other_geo_obj

    def restoration_of_normal_form(other_geo_obj, osm_combined_df, threshold=0.7) -> List[str]:
        """
        This function compares the extracted location entity with an OSM array and returns a normalized form if the percentage of similarity is at least 70%.
        """
        osm_name_obj = osm_combined_df['name'].tolist()
        similarity_matrix = np.zeros((len(other_geo_obj), len(osm_name_obj)))
        
        def extract_numbers(s):
            return re.findall(r'\d+', s)

        for i, word1 in enumerate(other_geo_obj):
            numbers_from_extraction = extract_numbers(word1)
            for j, word2 in enumerate(osm_name_obj):
                numbers_from_OSM_name = extract_numbers(word2)
                if numbers_from_extraction == numbers_from_OSM_name:
                    similarity = fuzz.ratio(word1, word2) / 100.0
                else:
                    similarity = 0  
                similarity_matrix[i, j] = similarity
        
        restoration_list = other_geo_obj.copy()
        for i in range(len(other_geo_obj)):
            max_index = np.argmax(similarity_matrix[i])
            if similarity_matrix[i, max_index] > threshold:
                restoration_list[i] = osm_name_obj[max_index]
            else:
                restoration_list[i] = ""
        
        return restoration_list
    
    @staticmethod
    def find_num_city_obj(text, NUM_CITY_OBJ) -> List[str]:
        """
        This function searches for urban objects in the text, the names of which are represented as a number. For example, "school No. 6".
        """
        text = text.lower()
        num_obj_list = []
        for key, forms in NUM_CITY_OBJ.items():
            for form in forms:
                pattern = fr'\b{re.escape(form)}\b\s+№?\s*(\d+)'
                matches = re.findall(pattern, text)
                for match in matches:
                    num_obj_list.append(f"{key} № {match}")
        num_obj_list = list(set(num_obj_list))
        num_obj_list_clear = {}
        for obj in num_obj_list:
            key = obj.split(' № ')[1]
            if key in num_obj_list_clear:
                if len(obj.split(' № ')[0]) > len(num_obj_list_clear[key].split(' № ')[0]):
                    num_obj_list_clear[key] = obj
            else:
                num_obj_list_clear[key] = obj
        
        return list(num_obj_list_clear.values())
    
    @staticmethod
    def combine_city_obj(df_obj) -> pd.DataFrame:
        """
        This function combines the found named urban objects and urban objects whose names are in the form of numbers.
        """
        df_obj['other_geo_obj'] = df_obj['other_geo_obj'] + df_obj['other_geo_obj_num']
        df_obj.drop(columns=['other_geo_obj_num'], inplace=True)
        return df_obj
    
    @staticmethod
    def expand_toponim(df_obj) -> pd.DataFrame:
        """
        This function splits the list of found entities into different rows for further analysis.
        """
        expanded_df = df_obj.copy()
        expanded_df['other_geo_obj'] = expanded_df['other_geo_obj'].apply(lambda x: x if isinstance(x, list) and x else None)
        expanded_df = expanded_df.explode('other_geo_obj').reset_index(drop=True)
        return expanded_df
    
    @staticmethod
    def find_geometry(toponim, osm_combined_df) -> List[str]:
        """
        This function finds the coordinate in the OSM array by the name of the city object.
        """
        if toponim is None:
            return None
        match = osm_combined_df[osm_combined_df['name'] == toponim]
        if not match.empty:
            geometry = match.iloc[0, 1]
            return geometry
        else:
            return None
    
    @staticmethod
    def find_geo_obj_tag(toponim, osm_combined_df) -> List[str]:
        """
        This function finds the geo_obj_tag in the OSM array by the name of the city object.
        """
        if toponim is None:
            return None
        match = osm_combined_df[osm_combined_df['name'] == toponim]
        if not match.empty:
            leisure = match.iloc[0, 2]
            return leisure
        else:
            return None

    
    @staticmethod
    def run(osm_city_name, df, text_column) -> pd.DataFrame:
        """
        This function launches the module for extracting urban objects from texts that do not relate to streets.
        """
        df_obj = df.copy()
        df_obj['Numbers'] = np.nan
        osm_combined_df = OtherGeoObjects.run_OSM_dfs(osm_city_name)
        logger.info('find_other_geo_obj started')
        df_obj['other_geo_obj'] = df_obj[text_column].apply(OtherGeoObjects.extract_geo_obj)
        df_obj['other_geo_obj_num'] = df_obj[text_column].apply(lambda x: OtherGeoObjects.find_num_city_obj(x, NUM_CITY_OBJ))
        df_obj = OtherGeoObjects.combine_city_obj(df_obj)
        df_obj['other_geo_obj'] = df_obj['other_geo_obj'].apply(lambda x: OtherGeoObjects.restoration_of_normal_form(x, osm_combined_df))
        df_obj = OtherGeoObjects.expand_toponim(df_obj)
        df_obj['geometry'] = df_obj['other_geo_obj'].apply(lambda x: OtherGeoObjects.find_geometry(x, osm_combined_df))
        df_obj['geo_obj_tag'] = df_obj['other_geo_obj'].apply(lambda x: OtherGeoObjects.find_geo_obj_tag(x, osm_combined_df))
        df_obj = df_obj[df_obj['geometry'].notna()]
        return df_obj
