"""
This module contains classes for retrieving and working with various types of data.

@class:GeoDataGetter:
This class is used to retrieve geospatial data from OpenStreetMap (OSM) based on given OSM ID and tags.

@class:VkCommentsParser:
A class for parsing and working with VK comments.

@class:Streets:
A class for working with street data.

@class:PostGetter:
A class used to retrieve post and comment data from the VK API.
"""
import osmnx as ox
import geopandas as gpd
import pandas as pd
from sloyka.src.constants import OSM_TAGS, GLOBAL_CRS, GLOBAL_METRIC_CRS, GLOBAL_EPSG
from shapely.ops import transform
from tqdm import tqdm
import requests
import sys
import datetime
import time
import osm2geojson
from typing import List

class GeoDataGetter:
    """
    This class is used to retrieve geospatial data from OpenStreetMap (OSM) based on given OSM ID and tags.

    Methods:
    - get_features_from_id: Retrieves features from the given OSM ID using the provided tags and OSM type, and returns the results as a GeoDataFrame.
    - _get_place_from_id: Retrieves the place from the given OSM ID and OSM type.
    - _process_tags: Processes the provided tags and returns a list of GeoDataFrames.
    - _get_features_from_place: Retrieves features from a specific place based on category and tag.
    - _handle_error: Handles any errors that occur during the process and prints an error message.
    """
    def get_features_from_id(
            self,
            osm_id: int,
            tags: dict,
            osm_type="R",
            selected_columns=['tag', 'element_type', 'osmid', 'name', 'geometry', 'centroid']
            ) -> gpd.GeoDataFrame:
        """
        Get features from the given OSM ID using the provided tags and OSM type, and return the results as a GeoDataFrame.
        
        Args:
            osm_id (int): The OpenStreetMap ID.
            tags (dict): The tags to filter by.
            osm_type (str, optional): The OpenStreetMap type. Defaults to "R".
            selected_columns (list, optional): The selected columns to include in the result GeoDataFrame. Defaults to ['tag', 'element_type', 'osmid', 'name', 'geometry', 'centroid'].
        
        Returns:
            gpd.GeoDataFrame: The GeoDataFrame containing the features.
        """
        place = self._get_place_from_id(osm_id, osm_type)
        gdf_list = self._process_tags(tags, place, selected_columns)
        
        if len(gdf_list) > 0:
            merged_gdf = pd.concat(gdf_list).reset_index().loc[:, selected_columns]
        else:
            merged_gdf = pd.DataFrame(columns=selected_columns)

        return merged_gdf

    def _get_place_from_id(self, osm_id, osm_type):
        place = ox.project_gdf(ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True))
        return place

    def _process_tags(self, tags, place, selected_columns):
        gdf_list = []
        place_name = place.name.iloc[0]
        for category, category_tags in tags.items():
            for tag in tqdm(category_tags, desc=f'Processing category {category}'):
                try:
                    gdf = self._get_features_from_place(place_name, category, tag)
                    gdf_list.append(gdf)
                except (AttributeError):
                    self._handle_error(category, tag)
                    pass
        return gdf_list

    def _get_features_from_place(self, place_name, category, tag):
        gdf = ox.features_from_place(place_name, tags={category: tag})
        gdf.geometry.dropna(inplace=True)
        gdf['tag'] = category
        gdf['centroid'] = gdf['geometry']

        tmpgdf = ox.projection.project_gdf(gdf, to_crs=GLOBAL_METRIC_CRS, to_latlong=False)
        tmpgdf['centroid'] = tmpgdf['geometry'].centroid
        tmpgdf = tmpgdf.to_crs(GLOBAL_CRS)
        gdf['centroid'] = tmpgdf['centroid']
        tmpgdf = None

        return gdf

    def _handle_error(self, category, tag):
        print(f'\nFailed to export {category}-{tag}\nException Info:\n{chr(10).join([str(line) for line in sys.exc_info()])}')

class VkCommentsParser:
    """
    A class for parsing and working with VK comments.

    Methods:
        - unix_to_date(ts): Convert a Unix timestamp to a date string.
        - nes_params(post_id, all_comments): Generate a dictionary of comments for a given post.
        - get_Comments(post_id, owner_id, token, nes_dict): Get comments from a VK post and parse them into a nested dictionary.
        - to_df(nes_dict): Create a pandas DataFrame from the given nested dictionary.
    """  
    def unix_to_date(ts):
        """
        Convert a Unix timestamp to a date string.

        Parameters:
            ts (int): The Unix timestamp to convert to a date string.

        Returns:
            str: The date string in the format 'YYYY-MM-DD'.
        """
        return datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')

    def nes_params(post_id, all_comments):
        """
        Generates a dictionary of comments for a given post, using the post ID and all comments.
        
        Parameters:
            post_id (int): The ID of the post for which comments are to be retrieved.
            all_comments (dict): A dictionary containing all comments, including profiles and items.
        
        Returns:
            dict: A dictionary containing the comments mapped to their IDs, along with associated profile information.
        """
        nes_dict = {}
        nes_dict = {}
        profiles = all_comments['profiles']
        comments = all_comments['items']
        first_string = ['NONE', 'NONE', 'NONE']
        for comment in comments:
            if len(comment['text']) > 3:
                second_string = [VkCommentsParser.unix_to_date(comment['date']), comment['likes']['count'],
                                comment['text']]
                for profile in profiles:
                    if comment['from_id'] == profile['id']:
                        first_string = [profile['first_name'], profile['last_name']]
                nes_dict[comment['id']] = first_string + second_string
        return nes_dict

    def get_Comments(post_id, owner_id, token, nes_dict={}): 
        """
        Get comments from a VK post and parse them into a nested dictionary.

        Args:
            post_id (int): The ID of the post.
            owner_id (int): The ID of the post owner.
            token (str): The access token for the VK API.
            nes_dict (dict): A nested dictionary to store the parsed comments. Defaults to an empty dictionary.

        Returns:
            dict: A nested dictionary containing the parsed comments.
        """
        version = 5.131
        offset = 0
        count = 100
        while offset < 500:
            response = requests.get('https://api.vk.com/method/wall.getComments',
                                params={
                                    'access_token': token,
                                    'v': version,
                                    'owner_id': owner_id,
                                    'post_id': post_id,
                                    'need_likes': 1,
                                    'count': count,
                                    'offset': offset,
                                    'extended': 1
                                }
                                )
            data_comments = response.json()['response']
            tempDict = VkCommentsParser.nes_params(post_id, data_comments)
            nes_dict.update(tempDict)
            offset += 100
            time.sleep(0.5)
        return nes_dict

    def to_df(nes_dict):
        """
        Create a pandas DataFrame from the given nested dictionary.

        Parameters:
            nes_dict (dict): The nested dictionary to be converted into a DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame created from the nested dictionary.
        """
        df = pd.DataFrame.from_dict(nes_dict, orient='index',
                                    columns=['name', 'last_name', 'date', 'likes', 'text', 'post_id'])
        return df

class Streets:
    """
    This class encapsulates functionality for retrieving street data
    for a specified city from OSM and processing it to extract useful
    information for geocoding purposes.
    """

    global_crs: int = 4326

    @staticmethod
    def get_city_bounds(
        osm_city_name: str, osm_city_level: int
    ) -> gpd.GeoDataFrame:
        """
        Method retrieves the boundary of a specified city from OSM
        using Overpass API and returns a GeoDataFrame representing
        the boundary as a polygon.
        """
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
                area[name="{osm_city_name}"]->.searchArea;
                (
                relation["admin_level"="{osm_city_level}"](area.searchArea);
                );
        out geom;
        """

        result = requests.get(
            overpass_url, params={"data": overpass_query}
        ).json()
        resp = osm2geojson.json2geojson(result)
        city_bounds = gpd.GeoDataFrame.from_features(resp["features"]).set_crs(
            Streets.global_crs
        )
        return city_bounds
    
class PostGetter:
    """
    A class used to retrieve post and comment data from the VK API.

    Methods:
    - get_group_post_ids(owner_id, your_token) -> List[int]: Retrieves a list of post IDs from a VK group.
    - unix_to_date(ts): Converts a Unix timestamp to a human-readable date format.
    - nes_params(post_id, all_comments): Processes necessary parameters from comments data.
    - get_Comments(post_id, owner_id, token): Retrieves comments for a specific post.
    - _to_df(nes_dict): Converts the processed dictionary data into a DataFrame.
    - run(your_owner_id, your_token, limit_posts=None): Runs the entire process to retrieve and process post and comment data.
    """
    def __init__():
        pass

    API_VERISON = 5.131
    OFFSET_STEP = 100
    OFFSET_LIMIT = 700
    COUNT_ITEMS = 100
    SLEEP_TIME = 0.5
    TIMEOUT_LIMIT = 15

    def get_group_post_ids(owner_id, your_token) -> List[int]:
        """
        Get a list of post IDs for a given owner ID using the VK API.

        Args:
            owner_id (int): The ID of the owner whose posts are being retrieved.
            your_token (str): The access token for making the API request.

        Returns:
            List[int]: A list of post IDs belonging to the specified owner ID.
        """
        offset = 0
        post_ids = []

        while offset < PostGetter.OFFSET_LIMIT:
            res = requests.get(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": your_token,
                    "v": PostGetter.API_VERISON,
                    "owner_id": owner_id,
                    "count": PostGetter.COUNT_ITEMS,
                    "offset": offset,
                },
            ).json()["response"]

            post_ids_new = [k["id"] for k in res["items"]]
            post_ids += post_ids_new
            offset += PostGetter.OFFSET_STEP

        return post_ids

    def unix_to_date(ts):
        """
        Convert a Unix timestamp to a date string.

        Parameters:
            ts (int): The Unix timestamp to be converted.

        Returns:
            str: The date string in the format "%Y-%m-%d".
        """
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    def nes_params(post_id, all_comments):
        """
        Creates a dictionary containing information about comments related to a specific post.

        Args:
            post_id: The ID of the post.
            all_comments: A dictionary containing all comments and profiles.

        Returns:
            nes_dict: A dictionary containing information about the comments.
        """
        nes_dict = {}
        profiles = all_comments["profiles"]
        comments = all_comments["items"]
        first_string = ["NONE", "NONE", "NONE"]
        for comment in comments:
            #TODO: refactor
            if len(comment["text"]) > 3: 
                second_string = [
                    PostGetter.unix_to_date(comment["date"]),
                    comment["text"],
                ]
                for profile in profiles:
                    #TODO: refactor
                    if comment["from_id"] == profile["id"]: 
                        first_string = [
                            profile["first_name"],
                            profile["last_name"],
                        ]
                #TODO: refactor
                nes_dict[comment["id"]] = (
                    first_string + second_string + [post_id]
                )
        return nes_dict

    def get_Comments(
        post_id, owner_id, token
    ):
        """
        Retrieves comments for a given post from the VK API.

        Args:
            post_id (int): The ID of the post for which comments are to be retrieved.
            owner_id (int): The ID of the owner of the post.
            token (str): The access token for making the API request.

        Returns:
            dict: A dictionary containing the retrieved comments.
        """
        temp_dict = {}
        offset = 0
        while offset < PostGetter.OFFSET_LIMIT:
            response = requests.get(
                "https://api.vk.com/method/wall.getComments",
                params={
                    "access_token": token,
                    "v": PostGetter.API_VERISON,
                    "owner_id": owner_id,
                    "post_id": post_id,
                    "count": PostGetter.COUNT_ITEMS,
                    "offset": offset,
                    "extended": 1,
                },
                timeout=PostGetter.TIMEOUT_LIMIT,
            )
            data_comments = response.json()["response"]
            comments_dict = PostGetter.nes_params(post_id, data_comments)
            temp_dict.update(comments_dict)
            
            offset += PostGetter.OFFSET_STEP
            time.sleep(PostGetter.SLEEP_TIME)
        return temp_dict
    
    def _to_df(nes_dict):  # перевод словаря в датафрейм
        #TODO: determine the cause of columns shift
        df = pd.DataFrame.from_dict(
            nes_dict,
            orient="index",
        )
        if 5 in df.columns:
            temp_df = df[df[0] == "NONE"]
            df = df.drop(temp_df.index)
            df = df.drop(columns=[5])
            temp_df = temp_df.drop(columns=[0])
            df.columns = ["name", "last_name", "date", "text", "post_id"]
            temp_df.columns = ["name", "last_name", "date", "text", "post_id"]
            df = pd.concat([df, temp_df]).sort_values(by='post_id')
        else:
            df.columns = ["name", "last_name", "date", "text", "post_id"]
        df.text = df["text"].str.replace("\n", " ")
        return df
    
    def run(your_owner_id, your_token, limit_posts=None):
        """
        Retrieves group post comments for a specified owner ID and token, up to a specified limit.
        
        Args:
            your_owner_id (int): The owner ID for the group.
            your_token (str): The token for authentication.
            limit_posts (int, optional): The maximum number of posts to retrieve comments for. Defaults to None.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the retrieved comments.
        """
        nes_dict = {}
        post_ids = PostGetter.get_group_post_ids(your_owner_id, your_token)
        for post_id in tqdm(post_ids[:limit_posts]):
            comments_dict = PostGetter.get_Comments(post_id, your_owner_id, your_token)
            nes_dict.update(comments_dict)
        
        return PostGetter._to_df(nes_dict)