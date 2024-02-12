from factfinder.src.semantic_model import Semantic_model, Visualiztor


# Инициация модели.
m = Semantic_model('data\\raw\posts_spb_today.csv', 'text')
model = m.make_model()

# Нахождение косинусной близости слов.
print(model.wv.similarity('полицейский', 'тротуар'))

# Сохранение модели.
m.save_model(model, 'semantic_models\\model.model')

# Тренировка существующей модели на новых данных.
trained_model = m.train_model(model_path='semantic_models\\model.model', 
                              training_data_path='data\\raw\\total_reports.csv',
                              column='Текст')

# Проверка результатов тренировки.
print(trained_model.wv.similarity('полицейский', 'тротуар'))

# Сохранение дотренированной модели.
m.save_model(model=trained_model, model_path='semantic_models\\trained_model.model')

# Построение графа знаний от указанного слова или списка слов.
Visualiztor(model_path='semantic_models\\trained_model.model', word='тротуар', depth=2, topn=10).save_graph_img(img_path='data/graph_1.jpg')
