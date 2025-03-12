import pytest
import torch
import pandas as pd
from soika import EmotionRecognizer

@pytest.fixture
def sample_dataframe():
    s_data = {'Текст комментария': {203: 'На Чайковского 63 тоже идет кап.ремонт. В квартире у пенсионеров побили стекла. Куда им обратиться?',
    204: 'Вся улица Жуковского и Восстания заклеена рекламой! Почему не действует полиция и администрация с ЖСК-1 ?'},
    'message_id': {203: 195, 204: 196}}
    return pd.DataFrame(s_data)
    
@pytest.fixture
def model():
    return EmotionRecognizer()

def test_emotion_recognizer(model, sample_dataframe):
    sample_dataframe["emotion"] = sample_dataframe["Текст комментария"].progress_map(lambda x: model.recognize_emotion(x))
    print(sample_dataframe)
    assert sample_dataframe.iloc[0]["emotion"] == "neutral"