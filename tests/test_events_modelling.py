import pytest
import torch
import geopandas as gpd
import pandas as pd
from shapely import Point
from sloyka import EventDetection

path_to_population = "data/raw/population.geojson"
path_to_data = "data/processed/messages.geojson"


@pytest.fixture
def gdf():
    gdf = gpd.read_file(path_to_data)
    gdf = gdf.head(6)
    return gdf


def test_event_detection(gdf):
    expected_name = "0_фурштатская_штукатурного слоя_слоя_отслоение"
    expected_risk = 0.405
    expected_messages = [4, 5, 3, 2]
    event_model = EventDetection()
    _, events, _ = event_model.run(
        gdf, path_to_population, "Санкт-Петербург", 32636, min_event_size=3
    )
    event_name = events.iloc[0]["name"]
    event_risk = events.iloc[0]["risk"].round(3)
    event_messages = [
        int(mid) for mid in events.iloc[0]["message_ids"].split(", ")
    ]
    assert event_name == expected_name
    assert event_risk == expected_risk
    assert all(mid in event_messages for mid in expected_messages)
