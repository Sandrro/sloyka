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
    def get_features_from_id(
        self,
        osm_id: int,
        tags: dict,
        osm_type="R",
        selected_columns=[
            "tag",
            "element_type",
            "osmid",
            "name",
            "geometry",
            "centroid",
        ],
    ) -> gpd.GeoDataFrame:
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
    def unix_to_date(ts):
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    def nes_params(post_id, all_comments):
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
        print(nes_dict)
        df = pd.DataFrame.from_dict(
            nes_dict,
            orient="index",
            columns=["name", "last_name", "date", "likes", "text", 'post_id'],
        )
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
    def __init__():
        pass

    API_VERISON = 5.131
    # OFFSET_STEP = 100
    # OFFSET_LIMIT = 700
    COUNT_ITEMS = 100
    SLEEP_TIME = 0.5
    TIMEOUT_LIMIT = 15

    def get_group_post_ids(owner_id, your_token, post_num_limit, step) -> List[int]:
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
                },
            ).json()["response"]

            post_ids_new = [k["id"] for k in res["items"]]
            post_ids += post_ids_new
            offset += step

        return post_ids
