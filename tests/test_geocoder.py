import pandas as pd
import pytest
from sloyka.src.geocoder.geocoder import Geocoder

@pytest.fixture
def sample_dataframe():
    s_data = {
        "text": [
            "На биржевой 15 снова шумят!!"
        ]
    }
    return pd.DataFrame(s_data)


def test_run_function(sample_dataframe):
    instance = Geocoder(osm_id=337422, city_tags = { "place": ["state"] })

    result_df = instance.run(df=sample_dataframe, group_column=None)

    print(result_df[['Street', 'Numbers']])

    assert result_df.loc[0, "Street"] == "биржевой"
    assert result_df.loc[0, "Numbers"] == "15"
