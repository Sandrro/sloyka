import csv
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gensim.models import word2vec


from paths import CUSTOM_MODELS_DIR_PATH, OUTPUT_DATA_DIR_PATH


class Visualiztor:
    '''
    Класс визуализирует данные модели в граф знаний на основе подающегося в него слова и наиболее близких к нему слов.
    '''
    def __init__(self, word: str, model_name: str='trained_model.model', depth: int=2, topn: int=10):
        # Функция инициализирует класс.

        self.word = word # Слово, от которого строится граф
        self.model_path = str(Path(CUSTOM_MODELS_DIR_PATH).joinpath(model_name)) # Путь к модели Word2Vec
        self.depth = depth # Глубина отрисовки графа
        self.topn = topn # Количество ближайших слов для поиска

    def wright_nodes_in_csv(self):
        # Функция записывает связанные слова в качестве узлов графа в файл с расширением .csv, а косинусную близоасть в качестве направления.

        # Загрузка модели 
        model = word2vec.Word2Vec.load(self.model_path)

        # Изолирование ключего слова
        center = [self.word]

        # Создание файла с расширением .csv для записи узлов и направлений
        with open(str(Path(OUTPUT_DATA_DIR_PATH).joinpath('graph.csv')), 'w+', encoding='utf-8') as file:

            writer = csv.writer(file, delimiter=';')
            writer.writerow(['subject', 'object', 'semantic_closeness'])

            
            new_list = []

            # Итерация глубины
            for i in range(self.depth):

                # Итерация количества связанных слов
                loads = len(center)
                for k in range(loads):
                    nodes = model.wv.most_similar(center[k], topn=self.topn)

                    # Запись каждого узла в файл
                    for j in nodes:
                        line = [center[k], j[0], j[1]]
                        writer.writerow(line)
                        new_list.append(j[0])

                    center = new_list
                    new_list = []

        print('Wrote csv.')

    def save_graph_img(self, img_name: str='kg1.jpg', options: dict={'node_color': 'yellow',     # Цвет узлов
                                                                    'node_size': 1000,          # Размер узлов
                                                                    'width': 1,                 # Ширина линий связи
                                                                    'arrowstyle': '-|>',        # Стиль стрелки для напрвленного графа
                                                                    'arrowsize': 18,            # Размер стрелки
                                                                    'edge_color':'blue',        # Цвет связи
                                                                    'font_size':20              # Размер шрифта
                                                                    }):
        # Функция создаёт и сохраняет граф в изображние на основе записанных данных.

        # Запись в файл узлов и направлений
        self.wright_nodes_in_csv()

        # Инициализация DataFrame
        columns = ['subject', 'object', 'semantic_closeness']
        df = pd.read_csv(str(Path(OUTPUT_DATA_DIR_PATH).joinpath('graph.csv')), delimiter=';', names=columns, encoding='utf-8')

        # Создание объекта для записи графа
        G=nx.from_pandas_edgelist(df,"subject","object", edge_attr=True, create_using=nx.MultiDiGraph())

        # Визуализация в изображение
        plt.figure(figsize=(30,30))
        pos = nx.spring_layout(G)
        
        # Визуализация графа в изображение и сохранение
        nx.draw(G, with_labels=True, pos = pos, **options)
        nx.draw_networkx_edge_labels(G, pos=pos)

        # Сохраняем картинку
        plt.savefig(img_name)

        print('Image is ready.')
                

            

    