import re
import pandas as pd
from rapidfuzz import fuzz
from nltk.stem.snowball import SnowballStemmer
from soika.src.utils.data_getter.historical_geo_data_getter import HistGeoDataGetter
from soika.src.utils.constants import (
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

    def get_osm_areas(self, place_name, tags={"boundary": "administrative", "place": True}):
        """
        Загружает и обрабатывает геометрии областей и населенных пунктов из OSM по названию территории с использованием osmnx.

        Args:
            place_name (str): Название территории (например, "Ленинградская область").
            tags (dict): Словарь тэгов OSM для фильтрации объектов (по умолчанию административные границы и place).

        Returns:
            GeoDataFrame: Обработанный GeoDataFrame с уникальными областями и геометриями.
        """
        try:
            gdf = ox.geometries_from_place(place_name, tags=tags)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки данных из OSM для '{place_name}': {e}")
        
        if gdf.empty:
            raise ValueError(f"Данные для территории '{place_name}' не найдены.")
        
        gdf = gdf.to_crs("EPSG:4326")
        
        columns_to_keep = ["geometry", "name", "place", "admin_level"]
        gdf.reset_index(drop=True, inplace=True)
        gdf = gdf[[col for col in columns_to_keep if col in gdf.columns]]

        if "admin_level" in gdf.columns:
            gdf["admin_level"].fillna(12, inplace=True)
            gdf["admin_level"] = gdf["admin_level"].astype(int)
            gdf = gdf[gdf["admin_level"] >= 4]
            gdf = gdf.sort_values("admin_level").drop_duplicates(subset=["name"], keep="first")
        
        gdf.reset_index(drop=True, inplace=True)
        gdf = gdf[gdf.geometry.geom_type != 'LineString']
        gdf["geometry"] = gdf.to_crs(4326).geometry.apply(
            lambda geom: geom.buffer(500) if geom.geom_type == "Point" else geom
        )

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

        return best_match
    
    def run(self, df, from_osm: bool = True, areas = None, place_name: str = None):
        df['processed_group_name'] = df.group_name.map(lambda x: self.preprocess_group_name(x))
        print('processed group names')
        if from_osm:
            df_areas = self.get_osm_areas(place_name)
        else:
            df_areas = areas
        df_areas = self.preprocess_area_names(df_areas)
        print('processed area names')
        df["best_match"] = df.apply(
            lambda row: pd.Series(self.match_group_to_area(row["processed_group_name"], df_areas)), axis=1
        )
        print('found matches')
        return df