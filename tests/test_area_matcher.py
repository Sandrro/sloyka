# test_area_matcher.py
import pytest
import pandas as pd
import re
from sloyka.src.area_matcher import AreaMatcher
from sloyka.src.data_getter import VKParser

@pytest.fixture
def test_df_groups():
    domain = 'pkio_klp'
    access_token = '96cbbc1496cbbc1496cbbc14b795dfa8b8996cb96cbbc14f2a8294fa4c4c6fc7e753a93'
    df_groups = VKParser().run_parser(domain, access_token, step=100, cutoff_date='2024-03-20', number_of_messages=10)
    return df_groups

def test_df_areas(test_df_groups: pd.DataFrame):
    osm_id = 337422
    area_matcher = AreaMatcher()
    test_df_groups['territory'] = test_df_groups['group_name'].apply(lambda x: area_matcher.match_area(x, osm_id))
    
    assert any(test_df_groups['territory'].apply(lambda x: bool(re.search(r'Колпин', x))))


