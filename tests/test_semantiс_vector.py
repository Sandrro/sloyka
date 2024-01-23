import os
from pathlib import Path

from semantic_vector.src.paths import CUSTOM_MODELS_DIR_PATH, SEMANTIC_VALUE_DIR_RATH
from semantic_vector.src.train_model import Custom_model
from semantic_vector.src.visualization import Visualiztor

def test_initialize_save_model():

    result = []
    m = Custom_model()
    # Инициация модели.
    model = m.make_model()
    similarity = model.wv.similarity(('полицейский', 'тротуар'))
    result.append(similarity)
    # Сохранение модели.
    m.save_model(model=model)
    model_path = str(Path(CUSTOM_MODELS_DIR_PATH).joinpath('model.model'))
    save_check = os.path.exists(model_path)
    result.append(save_check)

    assert result == [0.42077002, True]

def test_train_save_model():

    result = []
    m = Custom_model()
    # Тренировка существующей модели на новых данных.
    trained_model = m.train_model(model_name='model.model')
    similarity = trained_model.wv.similarity('полицейский', 'тротуар')
    result.append(similarity)

    # Сохранение дотренированной модели.
    m.save_model(model=trained_model, model_name='trained_model.model')
    trained_model_path = str(Path(CUSTOM_MODELS_DIR_PATH).joinpath('trained_model.model'))
    path_check = os.path.exists(trained_model_path)
    result.append(path_check)
    
    assert result == [0.18530989, True]

def test_visualize_graph():
    
    # Построение графа знаний от указанного слова или списка слов.
    Visualiztor(word='тротуар', depth=2, topn=10).save_graph_img()
    img_path = str(Path(SEMANTIC_VALUE_DIR_RATH).joinpath('kg1.jpg'))
    img_check = os.path.exists(img_path)

    assert img_check == True
    
