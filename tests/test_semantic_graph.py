import geopandas as gpd
import pytest
from sloyka import Semgraph

@pytest.fixture
def sample_data():
    gdf = gpd.read_parquet("sloyka\sample_data\sample_data_geocoded_emotioned.parquet")
    gdf['type'] = 'post'
    return gdf


def test_build_semantic_graph(sample_data):
    sm = Semgraph()
    G = sm.build_graph(sample_data,
                    id_column='message_id',
                    text_column='Текст комментария',
                    text_type_column="type",
                    toponym_column='full_street_name',
                    toponym_name_column='only_full_street_name',
                    toponym_type_column='Toponyms',
                    post_id_column="message_id",
                    parents_stack_column="message_id",
                    location_column='Location',
                    geometry_column='geometry')
    
    assert len(G.edges) == 88