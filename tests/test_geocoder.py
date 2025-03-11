import pandas as pd
import pytest
from soika.src.geocoder.geocoder import Geocoder

@pytest.fixture
def sample_dataframe():
    s_data = {'Текст комментария': {203: 'На Чайковского 63 тоже идет кап.ремонт. В квартире у пенсионеров побили стекла. Куда им обратиться?',
    204: 'Вся улица Жуковского и Восстания заклеена рекламой! Почему не действует полиция и администрация с ЖСК-1 ?'},
    'message_id': {203: 195, 204: 196}}
    return pd.DataFrame(s_data)


def test_run_function(sample_dataframe):
    osm_id = 337422 # Saint Petersburg
    geocoder = Geocoder(df=sample_dataframe, osm_id=osm_id, city_tags={'place':['state']}, text_column_name='Текст комментария')

    result = geocoder.run(group_column=None)

    print(result[['Street', 'Numbers']])

    assert result.loc[0, "Street"] == "чайковского"
    assert result.loc[0, "Numbers"] == "63"
