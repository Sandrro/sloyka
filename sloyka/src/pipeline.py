"""

This module contains classes for retrieving and working with various types of data.

"""
import networkx as nx
import geopandas as gpd
import pandas as pd
from sloyka.src.data_getter import VKParser
from sloyka.src.geocoder import Geocoder
from sloyka.src.semantic_graph import Semgraph
from pyvis.network import Network


class Pipeline:

    """
    A class for processing and analyzing social media data.
    
     Attributes:
    - domain (str): The domain of the social media owner.
    - token (str): The access token for the social media API.
    - count (int): The number of posts to retrieve.
    - date (str): The date from which to retrieve posts.
    - osm_city_name (str): The name of the city for geocoding.
    - osm_city_level (int): The level of detail for geocoding in the city.
    - posts (DataFrame): Contains the retrieved posts data.
    - res (DataFrame): Contains the geocoded data.
    - G (Graph): Represents the semantic graph.
    
    Methods:
    - get_posts(): Retrieves posts from VK social media using VKParser.
    - geocoding(): Performs geocoding on the posts data.
    - build_semantic_graph(): Builds a semantic graph based on geocoded data.
    - save_graphml(): Saves the semantic graph in GraphML format.
    - save_posts_csv(): Saves the retrieved posts data in a CSV file.
    - visualize_graph(): Visualizes the semantic graph in an HTML file.
    - process(): Executes the data processing pipeline.
    """

    def __init__(self, domain, token, count, date, osm_city_name, osm_city_level):
        self.domain = domain
        self.token = token
        self.count = count
        self.date = date
        self.osm_city_name = osm_city_name
        self.osm_city_level = osm_city_level
        self.posts = None
        self.res = None
        self.G = None

    def get_posts(self):
        self.posts = VKParser().run_parser(self.domain, self.token, self.count, self.date)

    def geocoding(self, text_column='text'):
        geocoder = Geocoder(osm_city_name=self.osm_city_name, osm_city_level=self.osm_city_level)
        self.res = geocoder.run(df=self.posts, text_column=text_column)

    def build_semantic_graph(self, device='cpu', key_score_filter=0.6, semantic_score_filter=0.75):
        sm = Semgraph(device=device)
        self.G = sm.build_graph(
            self.res,
            id_column='id',
            text_column='text',
            text_type_column='type',
            toponym_column='only_full_street_name',
            toponym_name_column='initial_street',
            toponym_type_column='Toponims',
            post_id_column='post_id',
            parents_stack_column='parents_stack',
            location_column='Location',
            geometry_column='geometry'
        )
        # key_score_filter=key_score_filter,
        # semantic_score_filter=semantic_score_filter)

    def save_graphml(self, filename='sem_graph.graphml'):
        if self.G:
            nx.write_graphml(self.G, filename, encoding='utf-8')

    def save_posts_csv(self, filename='data_posts.csv'):
        if self.posts is not None:
            self.posts.to_csv(filename, index=False)
        else:
            print("DataFrame не был создан. Пожалуйста, выполните сначала метод geocoding.")

    def visualize_graph(self, filename='graph.html'):
        if self.G:
            nt = Network('1000px', '1000px')
            nt.show_buttons(filter_=['physics'])
            nt.from_nx(self.G)
            nt.write_html(filename)
        else:
            print("Граф не был построен. Пожалуйста, выполните сначала метод build_semantic_graph.")

    def process(self):
        self.get_posts()
        self.save_posts_csv()
        self.geocoding()
        self.build_semantic_graph()
        self.save_graphml()
        self.visualize_graph()
        return self.posts, self.G