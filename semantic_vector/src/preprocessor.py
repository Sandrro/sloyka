import re
import pandas as pd
import nltk
import nltk.data
import pymorphy2
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from pathlib import Path

from paths import RAW_DATA_DIR_PATH


nltk.download('punkt')
nltk.download('stopwords')


class Preprocessor:
    '''
    Класс для предобработки данных из расширений файлов .csv и .txt.
    '''

    def __init__(self, filename: str, column: str='', delimeter: str=';'):
        # Функция инициализирует класс.

        self.file_path = str(Path(RAW_DATA_DIR_PATH).joinpath(filename)) # Путь к файлу
        self.tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle') # Получение токинизатора
        self.column = column # Наименование колонки для работы с файловыми расширениями .csv
        self.delimeter = delimeter # Разделитель для работы с файловыми расширениями .csv

    def review_to_word_list(self, review: str, remove_stopwords: bool=True) -> list:
        # Функция преобразует предложение в список слов и возвращает этот список.

        # Избавление от лишнего в данных
        review = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", review)

        review_text = BeautifulSoup(review, "lxml").get_text()
        review_text = re.sub("[^а-яА-Я]"," ", review_text)

        words = review_text.lower().split()

        if remove_stopwords:
            stops = stopwords.words("russian")
            words = [w for w in words if w not in stops]
        
        return words
    
    def review_to_sentence(self, review: str, remove_stopwords: bool=False) -> list:
        # Функция преобразует текст в список предложений и возвращает этот список.

        raw_sentensces = self.tokenizer.tokenize(review.strip())

        sentences = []

        for raw in raw_sentensces:

            if len(raw_sentensces) > 0:
                sentences.append(self.review_to_word_list(raw, remove_stopwords))

        return sentences
    
    def normolize_words(self, sentenses_list: list) -> list:
        #Функция приводит слова в начальную форму русского языка и возвращает двумерный список предложений.

        morph = pymorphy2.MorphAnalyzer()
        
        for i in sentenses_list:
            for j in i:

                form_list = morph.normal_forms(j)
                index = i.index(j)
                i[index] = form_list[0]

        return sentenses_list

    def clean_file(self) -> list:
        # Функция очищает файл с расширением .csv или .txt и приводит его в вид датасета для обучения модели, возвращает двумерный список.
        preprocessed_data = []

        # Если файл .csv
        if '.csv' in self.file_path:
            clean_sents = []
            
            # Инициплизация DataFrame
            data = pd.read_csv(self.file_path, delimiter=self.delimeter).dropna(subset=self.column)
            
            print("Parsing sentences from training set...")

            # Очистка текста
            for review in data[self.column]:
                clean_sents += self.review_to_sentence(review, self.tokenizer)

            preprocessed_data = self.normolize_words(clean_sents)

        # Если файл .txt
        elif '.txt' in self.file_path:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()

                text = re.sub('\n', ' ', text)
                sents = sent_tokenize(text)

                punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~„“«»†*—/\-‘’'
                clean_sents = []

                # Очистка текста
                print("Parsing sentences from training set...")

                for sent in sents:
                    s = [w.lower().strip(punct) for w in sent.split()]
                    clean_sents.append(s)

                preprocessed_data = self.normolize_words(clean_sents)
        
        # Если расширение файла неизвестно
        else:
            print('Datatype is not supported for preprocessing training data.\nPlease, use .csv or .txt file')

        return preprocessed_data


    