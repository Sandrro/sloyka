import pytest
import sys
import os

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)
from sloyka.src.data_getter import GeoDataGetter
from sloyka.src.constants import OSM_TAGS

tdict = [(category, category_tags) for category, category_tags in OSM_TAGS.items()]

@pytest.fixture
def geo_data_getter():
    return GeoDataGetter()

@pytest.mark.parametrize(
    "osm_id, tags",
    [
        (421007, {tdict[0][0]: tdict[0][1]}),
    ],
)
def test_process_data_and_assert(geo_data_getter, osm_id, tags):
    test_gdf = geo_data_getter.get_features_from_id(osm_id=osm_id, tags=tags)
    assert len(test_gdf) > 0
