# streets.py

import logging
import re
import requests
import osm2geojson
import geopandas as gpd
import networkx as nx
import pandas as pd
import osmnx as ox

import warnings

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

    global_crs: int = 4326
    logger = logging.getLogger(__name__)

    @staticmethod
    def get_city_bounds(osm_id: int) -> gpd.GeoDataFrame:
        """
        Method retrieves the boundary of a specified city from OSM
        using Overpass API and returns a GeoDataFrame representing
        the boundary as a polygon.
        """
        Streets.logger.info(f"Retrieving city bounds for osm_id {osm_id}")
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
            (
            relation({osm_id});
            );
            out geom;
            """

        try:
            result = requests.get(overpass_url, params={"data": overpass_query}).json()
            resp = osm2geojson.json2geojson(result)
            city_bounds = gpd.GeoDataFrame.from_features(resp["features"]).set_crs(Streets.global_crs)
            Streets.logger.debug(f"City bounds retrieved: {city_bounds}")
            return city_bounds
        except requests.exceptions.RequestException as e:
            Streets.logger.error(f"Error retrieving city bounds: {e}")
            raise StreetsError(f"Error retrieving city bounds: {e}")

    @staticmethod
    def get_drive_graph(city_bounds: gpd.GeoDataFrame) -> nx.MultiDiGraph:
        """
        Method uses the OSMnx library to retrieve the street network for a
        specified city and returns it as a NetworkX MultiDiGraph object, where
        each edge represents a street segment and each node represents
        an intersection.
        """
        Streets.logger.info("Retrieving drive graph")
        try:
            G_drive = ox.graph_from_polygon(city_bounds.dissolve()["geometry"].squeeze(), network_type="drive")
            Streets.logger.debug(f"Drive graph retrieved: {G_drive}")
            return G_drive
        except ox.exceptions.OSMnxError as e:
            Streets.logger.error(f"Error retrieving drive graph: {e}")
            raise StreetsError(f"Error retrieving drive graph: {e}")

    @staticmethod
    def graph_to_gdf(G_drive: nx.MultiDiGraph) -> gpd.GeoDataFrame:
        """
        Method converts the street network from a NetworkX MultiDiGraph object
        to a GeoDataFrame representing the edges (streets) with columns
        for street name, length, and geometry.
        """
        Streets.logger.info("Converting graph to GeoDataFrame")
        try:
            gdf = ox.graph_to_gdfs(G_drive, nodes=False)
            gdf["name"].dropna(inplace=True)
            gdf = gdf[["name", "length", "geometry"]]
            gdf.reset_index(inplace=True)
            gdf = gpd.GeoDataFrame(data=gdf, geometry="geometry")
            Streets.logger.debug(f"GeoDataFrame created: {gdf}")
            return gdf
        except ox.exceptions.OSMnxError as e:
            Streets.logger.error(f"Error converting graph to GeoDataFrame: {e}")
            raise StreetsError(f"Error converting graph to GeoDataFrame: {e}")

    @staticmethod
    def get_street_names(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Method extracts the unique street names from a
        GeoDataFrame of street segments.
        """
        Streets.logger.info("Extracting street names")
        names = set(gdf["name"].explode().dropna())
        df_streets = pd.DataFrame(names, columns=["street"])
        Streets.logger.debug(f"Street names extracted: {df_streets}")
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
        Streets.logger.debug(f"Finding toponim words in {x}")
        pattern = re.compile(
            r"путепровод|улица|набережная реки|проспект"
            r"|бульвар|мост|переулок|площадь|переулок"
            r"|набережная|канала|канал|дорога на|дорога в"
            r"|шоссе|аллея|проезд|линия",
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
        Streets.logger.debug(f"Dropping words from {x}")
        try:
            lst = re.split(
                r"путепровод|улица|набережная реки|проспект"
                r"|бульвар|мост|переулок|площадь|переулок"
                r"|набережная|канала|канал|дорога на|дорога в"
                r"|шоссе|аллея|проезд",
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
        Streets.logger.info("Clearing street names")
        streets_df["toponim_name"] = streets_df["street"].map(Streets.find_toponim_words_from_name)
        streets_df["street_name"] = streets_df["street"].map(Streets.drop_words_from_name)
        Streets.logger.debug(f"Street names cleared: {streets_df}")
        return streets_df

    @staticmethod
    def run(osm_id: int) -> pd.DataFrame:
        """
        A static method to run the process of getting street data based on the given
        OSM id, returning a pandas DataFrame.
        """
        Streets.logger.info(f"Running street data retrieval for osm_id {osm_id}")
        city_bounds = Streets.get_city_bounds(osm_id)
        streets_graph = Streets.get_drive_graph(city_bounds)
        streets_gdf = Streets.graph_to_gdf(streets_graph)
        streets_df = Streets.get_street_names(streets_gdf)
        streets_df = Streets.clear_names(streets_df)
        Streets.logger.info(f"Street data retrieval complete: {streets_df}")
        return streets_df
