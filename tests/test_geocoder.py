import pandas as pd
import pytest
from sloyka.src.geocoder.geocoder import Geocoder

@pytest.fixture
def sample_dataframe():
    s_data = {
        "text": [
            "На рубинштейна 3 снова шумят!!"
        ]
    }
    return pd.DataFrame(s_data)


def test_run_function(sample_dataframe):
    instance = Geocoder(df=sample_dataframe, osm_id=337422, city_tags = { "place": ["state"] }, text_column_name='text')

    result_df = instance.run(group_column=None)

    print(result_df[['Street', 'Numbers']])

    assert result_df.loc[0, "Street"] == "рубинштейна"
    assert result_df.loc[0, "Numbers"] == "3"