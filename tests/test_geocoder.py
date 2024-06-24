import pandas as pd
import pytest
from sloyka.src.geocoder.geocoder import Geocoder

@pytest.fixture
def sample_dataframe():
    s_data = {
        "text": [
            "Биржевая линия 16 дворовую территорию уберите, где работники?"
        ]
    }
    return pd.DataFrame(s_data)


def test_run_function(sample_dataframe):
    instance = Geocoder(osm_id=337422)

    result_df = instance.run(df=sample_dataframe)

    assert result_df.loc[0, "Street"] == "Биржевая"
    assert result_df.loc[0, "Numbers"] == "16"
