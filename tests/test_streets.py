import geopandas as gpd

from sloyka.src.geocoder import Streets


def test_get_city_bounds():
    result = Streets.get_city_bounds("Санкт-Петербург", 8)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.shape[0] != 0
    assert result.shape[1] == 4
