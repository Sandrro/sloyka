import pandas as pd
import pytest
from .src import Geocoder 
osm_city_level: int = 5,
osm_city_name: str = "Санкт-Петербург",

@pytest.fixture
def sample_dataframe(): 
    s_data = {"Текст комментария": ["Рубинштейна 25 дворовую территорию уберите, где работники?"]}
    return pd.DataFrame(s_data)

def test_run_function(sample_dataframe):

    instance = Geocoder(osm_city_name, osm_city_level)

    result_df = instance.run(sample_dataframe)

    assert result_df.loc[1, 'Street'] == "рубинштейна"
    assert result_df.loc[1, 'Numbers'] == "25"
