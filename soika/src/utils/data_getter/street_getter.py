import re
import geopandas as gpd
import pandas as pd

import warnings

from soika.src.utils.data_getter.geo_data_getter import GeoDataGetter as dg
from soika.src.utils.data_preprocessing import preprocessor as pp
from soika.src.utils.constants import TOPONYM_PATTERN

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class StreetsError(Exception):
    """Custom exception for streets-related errors"""

    pass


class Streets:
    """
    This class encapsulates functionality for retrieving street data
    for a specified city from OSM and processing it to extract useful
    information for geocoding purposes.
    """

    @staticmethod
    def get_street_names(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Method extracts the unique street names from a
        GeoDataFrame of street segments.
        """
        # Streets.logger.info("Extracting street names")
        names = set(gdf["name"].explode().dropna())
        df_streets = pd.DataFrame(names, columns=["street"])
        # Streets.logger.debug(f"Street names extracted: {df_streets}")
        return df_streets

    @staticmethod
    def find_toponim_words_from_name(x: str) -> str:
        """
        A method to find toponim words from the given name string.

        Args:
            x (str): The input name string.

        Returns:
            str: The found toponim word from the input name string, or None if not found.
        """
        # Streets.logger.debug(f"Finding toponim words in {x}")
        pattern = re.compile(
            TOPONYM_PATTERN,
            re.IGNORECASE,
        )

        match = pattern.search(x)

        if match:
            return match.group().strip().lower()
        else:
            return None

    @staticmethod
    def drop_words_from_name(x: str) -> str:
        """
        This function drops parts of street names that are not the name
        of the street (e.g. avenue).
        """
        # Streets.logger.debug(f"Dropping words from {x}")
        try:
            lst = re.split(
                TOPONYM_PATTERN,
                x,
            )
            lst.remove("")

            return lst[0].strip().lower()

        except ValueError:
            return x

    @staticmethod
    def clear_names(streets_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function pre-process the street names from the OSM.
        This step is necessary to match recognised street addresses later.
        We need to do this match because Nominatim is very sensitive geocoder
        and requires almost exact match between addresses in the OSM database
        and the geocoding address.
        """
        # Streets.logger.info("Clearing street names")
        streets_df["toponim_name"] = streets_df["street"].map(Streets.find_toponim_words_from_name)
        streets_df["street_name"] = streets_df["street"].map(Streets.drop_words_from_name)
        # Streets.logger.debug(f"Street names cleared: {streets_df}")
        return streets_df

    @staticmethod
    def run(osm_id: int) -> pd.DataFrame:
        """
        A static method to run the process of getting street data based on the given
        OSM id, returning a pandas DataFrame.
        """
        # Streets.logger.info(f"Running street data retrieval for osm_id {osm_id}")
        city_bounds = dg.get_city_bounds(osm_id)
        streets_graph = dg.get_drive_graph(city_bounds)
        streets_gdf = pp.graph_to_gdf(streets_graph)
        streets_df = Streets.get_street_names(streets_gdf)
        streets_df = Streets.clear_names(streets_df)
        # Streets.logger.info(f"Street data retrieval complete: {streets_df}")
        return streets_df
