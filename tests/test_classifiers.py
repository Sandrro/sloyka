import pytest
import torch
import pandas as pd
from sloyka.src.text_classifiers import TextClassifiers

path_to_file = "C:/Users/trolo/CodeNIRMA/sloyka/data/raw/Адмиралтейский.csv"

@pytest.fixture
def test_data():
    df_predict = pd.read_csv(path_to_file, sep=";")
    df_predict.rename(columns={"Текст комментария": "Текст"}, inplace=True)
    df_predict = df_predict.dropna(subset=["Текст"])
    df_predict = df_predict.head(3)
    return df_predict

@pytest.fixture
def model():
    return TextClassifiers(
        repository_id="Sandrro/text_to_subfunction_v10",
        number_of_categories=1,
        device_type=torch.device("cpu"),
    )

def test_init():
    # Arrange
    repository_id = "Sandrro/text_to_subfunction_v10"
    number_of_categories = 1
    device_type = torch.device("cpu")

    # Act
    classifier = TextClassifiers(
        repository_id, number_of_categories, device_type
    )

    # Assert
    assert classifier.REP_ID == repository_id
    assert classifier.CATS_NUM == number_of_categories
    assert classifier.classifier.model.name_or_path == repository_id
    assert (
        classifier.classifier.tokenizer.name_or_path
        == "cointegrated/rubert-tiny2"
    )

def test_cats_probs(model, test_data):
    expected_df = pd.DataFrame(
        {
            "cats": [
                "Вопросы граждан о проектах/планах/сроках/ходе проведения работ по благоустройству",
                "Не ЦУР",
                "Вопросы по оплате проезда в общественном транспорте",
            ],
            "probs": ["1.0", "0.999", "0.98"],
        }
    )

    test_data[["cats", "probs"]] = pd.DataFrame(
        test_data["Текст"].progress_map(lambda x: model.run_text_classifier_topics(x)).to_list()
    )
    assert test_data["cats"].equals(expected_df["cats"])
    assert test_data["probs"].equals(expected_df["probs"])

@pytest.fixture
def test_data():
    df_predict = pd.read_csv(path_to_file, sep=";")
    df_predict.rename(columns={"Текст комментария": "Текст"}, inplace=True)
    df_predict = df_predict.dropna(subset=["Текст"])
    df_predict = df_predict.head(3)
    return df_predict


#@pytest.fixture(name="model")
@pytest.fixture
def model():
    return TextClassifiers(
        repository_id="Sandrro/text_to_function_v2",
        number_of_categories=1,
        device_type=torch.device("cpu"),
    )
    
def test_init():
    # Arrange
    repository_id = "Sandrro/text_to_function_v2"
    number_of_categories = 1
    device_type = torch.device("cpu")

    # Act
    classifier = TextClassifiers(
        repository_id, number_of_categories, device_type
    )

    # Assert
    assert classifier.REP_ID == repository_id
    assert classifier.CATS_NUM == number_of_categories
    assert classifier.classifier.model.name_or_path == repository_id
    assert (
        classifier.classifier.tokenizer.name_or_path
        == "cointegrated/rubert-tiny2"
    )


@pytest.mark.parametrize(
    "text, expected_cats, expected_probs",
    [
        # Первый тестовый текст с одной категорией
        ("Хочу окунуться в это пространство.", "Благоустройство", "0.874")
    ],
)

# Определяем тестовую функцию
# Создаем объект класса TextClassifier с параметрами по умолчанию
def test_run(text, expected_cats, expected_probs, model):
    # Вызываем метод run с входным текстом и получаем список с категориями и вероятностями
    cats, probs = model.run_text_classifier(text)

    # Проверяем, что категории и вероятности равны ожидаемым
    assert cats == expected_cats
    assert probs == expected_probs


def test_cats_probs(model, test_data):
    expected_df = pd.DataFrame(
        {
            "cats": ["Благоустройство", "Другое", "Транспорт"],
            "probs": ["0.874", "0.538", "0.789"],
        }
    )

    test_data[["cats", "probs"]] = pd.DataFrame(
        test_data["Текст"].progress_map(lambda x: model.run_text_classifier(x)).to_list()
    )
    assert test_data["cats"].equals(expected_df["cats"])
    assert test_data["probs"].equals(expected_df["probs"])