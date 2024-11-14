import re
import requests
import pandas as pd
import geopandas as gpd
from rapidfuzz import fuzz, process
from nltk.stem.snowball import SnowballStemmer
from shapely.geometry import Polygon, Point, MultiPolygon
from sloyka.src.utils.data_getter.historical_geo_data_getter import HistGeoDataGetter
from sloyka.src.utils.constants import (
    AREA_STOPWORDS,
    GROUP_STOPWORDS,
    REGEX_PATTERN,
    REPLACEMENT_STRING,
)

class AreaMatcher:
    
    def __init__(self):
        self.area_cache = {}
        self.stemmer = SnowballStemmer("russian")

    def get_df_areas(self, osm_id, tags, date):
        if osm_id not in self.area_cache:
            geo_data_getter = HistGeoDataGetter()
            df_areas = geo_data_getter.get_features_from_id(osm_id=osm_id, tags=tags, date=date)
            df_areas = df_areas[df_areas["element_type"] != "way"]
            self.area_cache[osm_id] = df_areas
        return self.area_cache[osm_id]

    def get_osm_areas(self, osm_id, osm_type="relation"):
        """
        Загружает и обрабатывает геометрии областей и населенных пунктов из OSM для заданного OSM ID.
        Args:
            osm_id (int): Идентификатор объекта в OSM.
            osm_type (str): Тип объекта в OSM (по умолчанию "relation").
        Returns:
            GeoDataFrame: Обработанный GeoDataFrame с уникальными областями и геометриями.
        """
        query = f"""
            [out:json];
            {osm_type}({osm_id});
            map_to_area -> .a;
            (
            node(area.a)[place];
            way(area.a)[place];
            relation(area.a)[place];
            relation(area.a)[boundary=administrative];
            );
            out geom;
        """
        url = "http://overpass-api.de/api/interpreter"
        response = requests.get(url, params={'data': query})
        data = response.json()

        elements = data['elements']
        places = []

        for elem in elements:
            if 'tags' in elem:
                name = elem['tags'].get('name')
                place = elem['tags'].get('place')
                admin_level = elem['tags'].get('admin_level')
                
                if elem['type'] == 'node':
                    geometry = Point(elem['lon'], elem['lat'])
                elif elem['type'] == 'way' and 'geometry' in elem:
                    coords = [(point['lon'], point['lat']) for point in elem['geometry']]
                    geometry = Polygon(coords) if len(coords) >= 4 else Point(coords[0])
                elif elem['type'] == 'relation' and 'members' in elem:
                    polygons = []
                    for member in elem['members']:
                        if member['type'] == 'way' and 'geometry' in member:
                            coords = [(point['lon'], point['lat']) for point in member['geometry']]
                            if len(coords) >= 4:
                                polygons.append(Polygon(coords))
                            elif len(coords) == 1:
                                polygons.append(Point(coords[0]).buffer(100))
                    geometry = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0] if polygons else None

                if name and geometry:
                    places.append({
                        'name': name,
                        'place': place,
                        'admin_level': admin_level,
                        'geometry': geometry
                    })

        gdf = gpd.GeoDataFrame(places, geometry='geometry', crs="EPSG:4326")

        centroid = gdf.geometry.centroid.to_crs("EPSG:4326").unary_union.centroid
        utm_zone = int((centroid.x + 180) // 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone}"
        gdf = gdf.to_crs(utm_crs)

        gdf['geometry'] = gdf.apply(lambda row: row.geometry.buffer(100) 
                                    if row.geometry.geom_type == 'Point' or 
                                    (row.geometry.geom_type == 'Polygon' and len(row.geometry.exterior.coords) < 4) 
                                    else row.geometry, axis=1)

        gdf = gdf.to_crs("EPSG:4326")

        gdf.admin_level.fillna(12, inplace=True)
        gdf.admin_level = gdf.admin_level.astype(int)
        gdf = gdf.sort_values('admin_level').drop_duplicates(subset=['name'], keep='first')

        return gdf

    def preprocess_group_name(self, group_name):
        """
        Preprocesses a group name by converting it to lowercase, removing special characters, and removing specified stopwords.

        Args:
            group_name (str): The group name to preprocess.

        Returns:
            str: The preprocessed group name.
        """
        group_name = group_name.lower()
        group_name = re.sub(REGEX_PATTERN, REPLACEMENT_STRING, group_name)
        words_to_remove = GROUP_STOPWORDS
        for word in words_to_remove:
            group_name = re.sub(word, "", group_name, flags=re.IGNORECASE)
        return group_name

    def preprocess_area_names(self, df_areas):
        """
        Preprocesses the area names in the given DataFrame by removing specified stopwords, converting the names to lowercase,
        and stemming them.

        Parameters:
            df_areas (DataFrame): The DataFrame containing the area names.

        Returns:
            DataFrame: The DataFrame with preprocessed area names, where the 'area_name' column contains the original names
                      with stopwords removed, the 'area_name_processed' column contains the lowercase names with special characters
                      removed, and the 'area_stems' column contains the stemmed names.
        """
        words_to_remove = AREA_STOPWORDS
        for word in words_to_remove:
            df_areas["area_name"] = df_areas["name"].str.replace(word, "", regex=True)

        df_areas["area_name_processed"] = df_areas["area_name"].str.lower()
        df_areas["area_name_processed"] = df_areas["area_name_processed"].str.replace(
            REGEX_PATTERN, REPLACEMENT_STRING, regex=True
        )
        df_areas["area_stems"] = df_areas["area_name_processed"].apply(
            lambda x: [self.stemmer.stem(word) for word in x.split()]
        )
        return df_areas

    def match_group_to_area(self, group_name, df_areas):
        """
        Matches a given group name to an area in a DataFrame of areas.

        Args:
            group_name (str): The name of the group to match.
            df_areas (DataFrame): The DataFrame containing the areas to match against.

        Returns:
            tuple: A tuple containing the best match for the group name and the admin level of the match.
                   If no match is found, returns (None, None).
        """
        group_name_stems = [self.stemmer.stem(word) for word in group_name.split()]
        max_partial_ratio = 20
        max_token_sort_ratio = 20
        best_match = None
        admin_level = None

        for _, row in df_areas.iterrows():
            area_stems = row["area_stems"]

            partial_ratio = fuzz.partial_ratio(group_name, row["area_name_processed"])
            token_sort_ratio = fuzz.token_sort_ratio(group_name_stems, area_stems)

            if partial_ratio > max_partial_ratio and token_sort_ratio > max_token_sort_ratio:
                max_partial_ratio = partial_ratio
                max_token_sort_ratio = token_sort_ratio
                best_match = row["area_name"]
                admin_level = row["admin_level"]

        return best_match, admin_level
    
    def run(self, df, osm_id):
        df['processed_group_name'] = df.group_name.map(lambda x: self.preprocess_group_name(x))
        df_areas = self.get_osm_areas(osm_id)
        df_areas = self.preprocess_area_names(df_areas)
        df[["best_match", "admin_level"]] = df.apply(
            lambda row: pd.Series(self.match_group_to_area(row["processed_group_name"], df_areas)), axis=1
        )
        return df