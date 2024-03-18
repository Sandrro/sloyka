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
from sloyka.src.constants import (
    OSM_TAGS,
    GLOBAL_CRS,
    GLOBAL_METRIC_CRS,
    GLOBAL_EPSG,
)
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
            merged_gdf = (
                pd.concat(gdf_list).reset_index().loc[:, selected_columns]
            )
        else:
            merged_gdf = pd.DataFrame(columns=selected_columns)

        return merged_gdf

    def _get_place_from_id(self, osm_id, osm_type):
        place = ox.project_gdf(
            ox.geocode_to_gdf(osm_type + str(osm_id), by_osmid=True)
        )
        return place

    def _process_tags(self, tags, place, selected_columns):
        gdf_list = []
        place_name = place.name.iloc[0]
        for category, category_tags in tags.items():
            for tag in tqdm(
                category_tags, desc=f"Processing category {category}"
            ):
                try:
                    gdf = self._get_features_from_place(
                        place_name, category, tag
                    )
                    gdf_list.append(gdf)
                except AttributeError:
                    self._handle_error(category, tag)
                    pass
        return gdf_list

    def _get_features_from_place(self, place_name, category, tag):
        gdf = ox.features_from_place(place_name, tags={category: tag})
        gdf.geometry.dropna(inplace=True)
        gdf["tag"] = category
        gdf["centroid"] = gdf["geometry"]

        tmpgdf = ox.projection.project_gdf(
            gdf, to_crs=GLOBAL_METRIC_CRS, to_latlong=False
        )
        tmpgdf["centroid"] = tmpgdf["geometry"].centroid
        tmpgdf = tmpgdf.to_crs(GLOBAL_CRS)
        gdf["centroid"] = tmpgdf["centroid"]
        tmpgdf = None

        return gdf

    def _handle_error(self, category, tag):
        print(
            f"\nFailed to export {category}-{tag}\nException Info:\n{chr(10).join([str(line) for line in sys.exc_info()])}"
        )


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
        profiles = all_comments["profiles"]
        comments = all_comments["items"]
        first_string = ["NONE", "NONE"]
        for comment in comments:
            if len(comment["text"]) > 3:
                second_string = [
                    VkCommentsParser.unix_to_date(comment["date"]),
                    comment["likes"]["count"],
                    comment["text"],
                ]
                for profile in profiles:
                    if comment["from_id"] == profile["id"]:
                        first_string = [
                            profile["first_name"],
                            profile["last_name"],
                        ]
                nes_dict[comment["id"]] = first_string + second_string
                nes_dict[comment["id"]] += [post_id]
        # print(nes_dict)
        return nes_dict

    def get_Comments(
        post_ids: list,
        owner_id: str,
        token: str,
        comments_num=1000,
        offset_limit=500,
        offset_step=100,
    ):
        version = 5.131
        nes_dict={}
        
        
        for post_id in tqdm(post_ids):
            offset = 0
            while offset < offset_limit:
                response = requests.get(
                    "https://api.vk.com/method/wall.getComments",
                    params={
                        "access_token": token,
                        "v": version,
                        "owner_id": owner_id,
                        "post_id": post_id,
                        "need_likes": 1,
                        "count": offset_step,
                        "offset": offset,
                        "extended": 1,
                    },
                )
                if response.ok:
                    data_comments = response.json()["response"]
                    # print(data_comments)
                    tempDict = VkCommentsParser.nes_params(
                        post_id, data_comments
                    )
                    
                    # print(len(nes_dict), tempDict)
                    if len(tempDict) > 0:
                        nes_dict.update(tempDict)
                else:
                    print(response.status_code)

                if len(nes_dict) >= comments_num:
                    # print(len(nes_dict))
                    return VkCommentsParser.to_df(nes_dict)

                offset += offset_step
                # print(offset)
                time.sleep(0.5)
            
        return VkCommentsParser.to_df(nes_dict)

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


class VkPostGetter:

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

    API_VERISON = 5.131
    # OFFSET_STEP = 100
    # OFFSET_LIMIT = 700
    COUNT_ITEMS = 100
    SLEEP_TIME = 0.5
    TIMEOUT_LIMIT = 15

    def get_group_post_ids(owner_id, your_token, post_num_limit, step) -> List[int]:
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

        while offset < post_num_limit:
            res = requests.get(
                "https://api.vk.com/method/wall.get",
                params={
                    "access_token": your_token,
                    "v": VkPostGetter.API_VERISON,
                    "owner_id": owner_id,
                    "count": step,
                    "offset": offset,
                }, timeout=90
            ).json()["response"]

            post_ids_new = [k["id"] for k in res["items"]]
            post_ids += post_ids_new
            offset += step

        return post_ids

    def run(owner_id:str, your_token:str,  step:int, cutoff_date:str, number_of_messages:int = float('inf')):
        token = your_token
        domain = owner_id
        version = 5.131
        offset = 0
        all_posts = []
        if step > number_of_messages:
            step = number_of_messages
        while offset < number_of_messages:
            response = requests.get('https://api.vk.com/method/wall.get',
                                    params={
                                        'access_token': token,
                                        'v': version,
                                        'domain': domain,
                                        'count': step,
                                        'offset': offset
                                    }
                                    )
            data = response.json()['response']['items']
            offset += step
            current_posts = pd.json_normalize(data)
            current_posts = current_posts[['date', 'id', 'text', 'views.count', 'likes.count', 'reposts.count']]
            current_posts['date'] = [datetime.datetime.fromtimestamp(current_posts['date'][i]) for i in range(len(current_posts['date']))]
            all_posts.append(current_posts)
            print(current_posts.date.min())
            if any(current_posts['date'] < datetime.datetime.strptime(cutoff_date, '%Y-%m-%d')):
                print('finished')
                break
            time.sleep(0.5)
        df_posts = pd.concat(all_posts).reset_index(drop=True)
        df_posts = df_posts[df_posts.text.map(lambda x: len(x)) > 0]
        df_posts['text'] = df_posts['text'].str.replace(r'\n', '', regex=True)
        df_posts['link'] = df_posts['text'].str.extract(r'(https://\S+)')
        return df_posts
    
    class CommentsReply:
        def get_comments(self,owner_id, post_id, access_token):
            params = {
                'owner_id': owner_id,
                'post_id': post_id,
                'access_token': access_token,
                'v': '5.131',
                'extended': 1,
                'count': 100,
                'need_likes': 1
            }

            comments = []

            response = requests.get('https://api.vk.com/method/wall.getComments', params=params)
            data = response.json()

            if 'response' in data:
                for item in data['response']['items']:
                    item['date'] = datetime.utcfromtimestamp(item['date']).strftime('%Y-%m-%d %H:%M:%S')
                    if 'likes' in item:
                        item['likes_count'] = item['likes']['count']
                    comments.append(item)
                    if item['thread']['count'] > 0:
                        params['comment_id'] = item['id']
                        subcomments = self.get_subcomments(owner_id, post_id, access_token, params)
                        comments.extend(subcomments)

            return comments

        def get_subcomments(self, owner_id, post_id, access_token, params):
            subcomments = []

            response = requests.get('https://api.vk.com/method/wall.getComments', params=params)
            data = response.json()

            if 'response' in data:
                for item in data['response']['items']:
                    item['date'] = datetime.utcfromtimestamp(item['date']).strftime('%Y-%m-%d %H:%M:%S')
                    if 'likes' in item:
                        item['likes_count'] = item['likes']['count']
                    subcomments.append(item)

            return subcomments

        def comments_to_dataframe(self, comments):
            df = pd.DataFrame(comments)
            df = df[['id', 'date', 'text', 'post_id', 'parents_stack', 'likes_count']]
            return df
        
        def run(self, owner_id, post_ids, access_token):
            all_comments = []
            for post_id in post_ids:
                comments = self.get_comments(owner_id, post_id, access_token)
                all_comments.extend(comments)
            df = self.comments_to_dataframe(all_comments)
            return df