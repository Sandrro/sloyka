from gensim.models import word2vec
from pathlib import Path

from preprocessor import Preprocessor
import paths


class Custom_model:
    '''
    Основной класс библиотеки, создающий и тренирующий модели.
    '''
    def __init__(self, file_name: str='posts_spb_today.csv', column:str='text', workers: int=4, min_count: int=2, window:int=10, sample: float=1e-3):
        '''Функция инициализирует класс.'''
        
        self.file_name = file_name # Имя файла
        self. workers = workers # Число потоков
        self.min_count = min_count # Минимальное количество повторения слов для вхождения в корпус модели
        self.window = window # Окно наблюдений
        self.sample = sample # Размер downsampling'a частовстречающихся слов
        self.column = column # Наименование колонки для работы с файловыми расширениями .csv

    def make_model(self) -> word2vec.Word2Vec:
        '''Функция создаёт и возвращает модель класса Word2Vec, необходимо сохранять в переменную.'''

        training_data = Preprocessor(self.file_name, self.column).clean_file()
        model = word2vec.Word2Vec(training_data, workers=self.workers, min_count=self.min_count, window=self.window, sample=self.sample)

        return model
    
    def save_model(self, model: word2vec.Word2Vec, model_name:str='model.model') -> None:
        '''Функция сохраняеи модель в небинарный файл.'''

        model_path = str(Path(paths.CUSTOM_MODELS_DIR_PATH).joinpath(model_name))

        model.save(model_path)

        print(f'Model saved as {model_name} in {paths.CUSTOM_MODELS_DIR_PATH}')

    def train_model(self, model_name: str='model.model', training_data_name: str='total_reports.csv', column: str='Текст', epochs: int=5) -> word2vec.Word2Vec:
        '''Функция тренирует предобученную пользовательскую модель и возвращает новую, необходимо записывать в переменную.'''

        model_path = str(Path(paths.CUSTOM_MODELS_DIR_PATH).joinpath(model_name))

        # Загрузка модели
        model = word2vec.Word2Vec.load(model_path)

        # Предобработка данных
        training_data = Preprocessor(training_data_name, column).clean_file()

        # Создание словаря модели и тренировка на датасете
        model.build_vocab(training_data, update=True)
        model.train(training_data, total_examples=model.corpus_count, epochs=epochs)

        return model

