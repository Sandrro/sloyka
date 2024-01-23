from paths import RAW_DATA_DIR_PATH
from train_model import Custom_model
from visualization import Visualiztor


# Инициация модели.
model = Custom_model().make_model()

# Нахождение косинусной близости слов.
print(model.wv.similarity('полицейский', 'тротуар'))

# Сохранение модели.
m = Custom_model()
m.save_model(model=model)

# Тренировка существующей модели на новых данных.
trained_model = m.train_model(model_name='model.model')

# Проверка результатов тренировки.
print(trained_model.wv.similarity('полицейский', 'тротуар'))

# Сохранение дотренированной модели.
m.save_model(model=trained_model, model_name='trained_model.model')

# Построение графа знаний от указанного слова или списка слов.
Visualiztor(word='тротуар', depth=2, topn=10).save_graph_img()
