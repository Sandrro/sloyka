import pytest
import pandas as pd
from soika import City_services

@pytest.fixture
def sample_dataframe():
    s_data = {'Текст комментария': {203: 'Когда уже на Юго западе будет метро? Весь день в пути проводим!',
    204: 'Вся улица Жуковского и Восстания заклеена рекламой! Почему не действует полиция и администрация с ЖСК-1 ?'},
    'message_id': {203: 195, 204: 196}}
    return pd.DataFrame(s_data)

@pytest.fixture
def model():
    return City_services()

def test_services(model, sample_dataframe):
    result = model.run(sample_dataframe, "Текст комментария")
    print(result)
    assert result.iloc[0]["City_services"][0] == "Метро"