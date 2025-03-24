"""
TODO: add spellchecker since there might be misspelled words.

This module is aimed to provide necessary tools to find mentioned
location in the text. 

@class:Location:
A class aimed to efficiently geocode addresses using Nominatim. 
Geocoded addresses are stored in the 'book' dictionary argument. 
Thus, if the address repeats, it would be taken from the book.

@class:Streets: 
A class encapsulating functionality for retrieving street data
for a specified city from OSM and processing it to extract useful information
for geocoding purposes.

@class:Geocoder:
A class providing functionality for simple geocoding and address extraction.
"""

import re
import flair
from soika.src.utils.constants import GLOBAL_CRS
import geopandas as gpd
import pandas as pd
import pymorphy2
import torch
import math
from rapidfuzz import fuzz
from nltk.stem.snowball import SnowballStemmer
from soika.src.utils.data_getter.historical_geo_data_getter import HistGeoDataGetter
from soika.src.utils.constants import (
    AREA_STOPWORDS,
    GROUP_STOPWORDS,
    REGEX_PATTERN,
    REPLACEMENT_STRING,
)

from flair.models import SequenceTagger
from shapely.geometry import Point
from tqdm import tqdm

from loguru import logger

from pandarallel import pandarallel
from soika.src.geocoder.city_objects_extractor import OtherGeoObjects
from soika.src.utils.data_getter.street_getter import Streets
from soika.src.utils.data_getter.location_getter import Location
from soika.src.utils.data_getter.geo_data_getter import GeoDataGetter
from soika.src.utils.data_processing.area_matcher import AreaMatcher
from soika.src.utils.data_preprocessing.preprocessor import PreprocessorInput
from soika.src.geocoder.street_extractor import StreetExtractor
from soika.src.geocoder.word_form_matcher import WordFormFinder

tqdm.pandas()


class Geocoder:
    """
    This class provides functionality of a simple geocoder.
    """
    pandarallel_initialized = False
    def __init__(
        self,
        df,
        model_path: str = "Geor111y/flair-ner-addresses-extractor",
        device: str = "cpu",
        territory_name: str = None,
        osm_id: int = None,
        city_tags: dict = {"place": ["state"]},
        stemmer_lang: str = "russian",
        text_column_name: str = 'text',
        nb_workers: int = -1
    ):
        if not Geocoder.pandarallel_initialized:
            pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)
            Geocoder.pandarallel_initialized = True

        self.text_column_name = text_column_name
        self.df = PreprocessorInput().run(df, text_column_name)
        self.device = device
        flair.device = torch.device(device)
        self.classifier = SequenceTagger.load(model_path)
        self.osm_id = osm_id
        if territory_name:
            self.osm_city_name = territory_name
        else:
            self.osm_city_name = (
                GeoDataGetter()
                .get_features_from_id(osm_id=self.osm_id, tags=city_tags, selected_columns=["name", "geometry"])
                .iloc[0]["name"]
            )
        self.street_names = Streets.run(self.osm_id)
        self.stemmer = SnowballStemmer(stemmer_lang)

    @staticmethod
    def get_stem(street_names_df: pd.DataFrame) -> pd.DataFrame:
        """
        Function finds the stem of the word to find this stem in the street
        names dictionary (df).
        """
        logger.info("get_stem started")

        morph = pymorphy2.MorphAnalyzer()
        cases = ["nomn", "gent", "datv", "accs", "ablt", "loct"]

        for case in cases:
            street_names_df[case] = street_names_df["street_name"].apply(
                lambda x: morph.parse(x)[0].inflect({case}).word if morph.parse(x)[0].inflect({case}) else None
            )
        return street_names_df


    @staticmethod
    def get_level(row: pd.Series) -> str:
        """
        Addresses in the messages are recognized on different scales:
        1. Where we know the street name and house number -- house level;
        2. Where we know only street name -- street level (with the centroid
        geometry of the street);
        3. Where we don't know any info but the city -- global level.
        """

        if (not pd.isna(row["Street"])) and (row["Numbers"] == ""):
            return "street"
        elif (not pd.isna(row["Street"])) and (row["Numbers"] != ""):
            return "house"
        else:
            return "global"

 
    def create_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Function simply creates gdf from the recognised geocoded geometries.
        """
        logger.info("create_gdf started")

        df["Location"] = df["addr_to_geocode"].progress_apply(Location().query)
        df = df.dropna(subset=["Location"])
        df["geometry"] = df['Location'].apply(lambda x: Point(x.longitude, x.latitude))
        df["Location"] = df['Location'].apply(lambda x: x.address)
        df["Numbers"].astype(str)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=GLOBAL_CRS)

        return gdf

    def set_global_repr_point(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        This function set the centroid (actually, representative point) of the
        geocoded addresses to those texts that weren't geocoded (or didn't
        contain any addresses according to the trained NER model).
        """

        try:
            gdf.loc[gdf["level"] == "global", "geometry"] = gdf.loc[
                gdf["level"] != "global", "geometry"
            ].unary_union.representative_point()
        except AttributeError:
            pass

        return gdf

    def merge_to_initial_df(self, gdf: gpd.GeoDataFrame, initial_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        This function merges geocoded df to the initial df in order to keep
        all original attributes.
        """

        # initial_df.drop(columns=['key_0'], inplace=True)
        gdf = initial_df.join(
            gdf[
                [
                    "Street",
                    "initial_street",
                    "only_full_street_name",
                    "Numbers",
                    "Score",
                    "location_options",
                    "Location",
                    "geometry",
                ]
            ],
            how="outer",
        )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=GLOBAL_CRS)

        return

    def assign_street(variable):
        '''Simple workaround'''

        if isinstance(variable, float) and math.isnan(variable):
            return "street"
        return variable

    def get_df_areas(self, osm_id, tags):
        """
        Retrieves the GeoDataFrame of areas corresponding to the given OSM ID and tags.

        Args:
            osm_id (int): The OpenStreetMap ID.
            tags (dict): The tags to filter by.
            date (str): The date of the data to retrieve.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame containing the areas.

        This function first checks if the GeoDataFrame corresponding to the given OSM ID is already in the cache.
        If it is, it returns the cached GeoDataFrame. Otherwise, it retrieves the GeoDataFrame from the HistGeoDataGetter,
        filters out the 'way' elements, and adds it to the cache. Finally, it returns the GeoDataFrame from the cache.
        """
        area_cache = {}
        if osm_id not in area_cache:
            geo_data_getter = HistGeoDataGetter()
            df_areas = geo_data_getter.get_features_from_id(osm_id=osm_id, tags=tags)
            df_areas = df_areas[df_areas["element_type"] != "way"]
            area_cache[osm_id] = df_areas
        return area_cache[osm_id]

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
                admin_level = row["key"]

        return best_match, admin_level

    def run(
        self, df: pd.DataFrame = None, tags:dict|None=None, 
        group_column: str | None = "group_name", search_for_objects=False
    ):
        """
        Runs the data processing pipeline on the input DataFrame.

        Args:
            tags (dict): The tags to filter by.
            date (str): The date of the data to retrieve.
            df (pd.DataFrame): The input DataFrame.
            text_column (str, optional): The name of the text column in the DataFrame. Defaults to "text".

        Returns:
            gpd.GeoDataFrame: The processed DataFrame after running the data processing pipeline.

        This function retrieves the GeoDataFrame of areas corresponding to the given OSM ID and tags.
        It then preprocesses the area names and matches each group name to an area. The best match
        and admin level are assigned to the DataFrame. The function also retrieves other geographic
        objects and street names, preprocesses the street names, finds the word form, creates a GeoDataFrame,
        merges it with the other geographic objects, assigns the street tag, and returns the final GeoDataFrame.
        """
        df = self.df
        initial_df = df.copy()
        text_column = self.text_column_name

        if search_for_objects:
            df_obj = OtherGeoObjects.run(self.osm_id, df, text_column)
            

        if group_column:
            df = AreaMatcher.run(self.osm_id)

        df[text_column] = df[text_column].astype(str).str.replace('\n', ' ')
        df[text_column] = df[text_column].apply(str)
        

        df = StreetExtractor.process_pipeline(df, text_column, self.classifier)
        street_names = self.get_stem(self.street_names)

        df = WordFormFinder(self.osm_city_name).find_word_form(df, street_names)

        del street_names
        gdf = self.create_gdf(df)

        if search_for_objects:
            gdf = pd.concat([gdf, df_obj], ignore_index=True)
            del df_obj
            gdf["geo_obj_tag"] = gdf["geo_obj_tag"].apply(Geocoder.assign_street)

        gdf = pd.merge(gdf, initial_df, on=text_column, how='right')

        gdf.set_crs(GLOBAL_CRS, inplace=True)
        return gdf
    