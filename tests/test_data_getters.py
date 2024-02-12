import pytest
import geopandas as gpd
from sloyka.src.data_getter import VkCommentsParser
from sloyka.src.data_getter import Streets
from sloyka.src.data_getter import GeoDataGetter
from sloyka.src.constants import OSM_TAGS

tdict = [(category, category_tags) for category, category_tags in OSM_TAGS.items()]

@pytest.mark.parametrize(
        "post_id, owner_id, token, result_len",
        [
            ("10796", -51988192, '96cbbc1496cbbc1496cbbc14b795dfa8b8996cb96cbbc14f2a8294fa4c4c6fc7e753a93', 2)
        ],
)

def test_get_comments(post_id, owner_id, token, result_len):
    test_df = VkCommentsParser.get_Comments(post_id, owner_id, token)
    assert len(test_df) == result_len

def test_get_city_bounds():
    result = Streets.get_city_bounds("Санкт-Петербург", 8)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.shape[0] != 0
    assert result.shape[1] == 4

@pytest.fixture
def geo_data_getter():
    return GeoDataGetter()

@pytest.mark.parametrize(
    "osm_id, tags, result_len",
    [
        (421007, {tdict[0][0]: tdict[0][1]}, 305),
    ],
)
def test_process_data_and_assert(geo_data_getter, osm_id, tags, result_len):
    test_gdf = geo_data_getter.get_features_from_id(osm_id=osm_id, tags=tags)
    assert len(test_gdf) == result_len
