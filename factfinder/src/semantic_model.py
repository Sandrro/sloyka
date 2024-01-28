'''
This module is aimed to provide necessary tools to create a semantic graph from texts. 
In this scenario texts are comments in social networks (e.g. Vkontakte).
Thus the model was trained on the corpus of comments on Russian language.
'''

import matplotlib.pyplot as plt
import networkx as nx
import nltk
from bs4 import BeautifulSoup
from gensim.models import word2vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import pandas as pd
import pymorphy2
import re


nltk.download('punkt')
nltk.download('stopwords')


class Preprocessor:
    '''
    A class for preprocessing data from .csv and .txt file extensions.
    '''

    def __init__(self, filepath: str, column: str='', delimeter: str=';') -> None:
        '''The function initialises the class.'''

        self.file_path = filepath # File path
        self.tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle') # Receipt of the tokeniser
        self.column = column # Column name for working with .csv file extensions
        self.delimeter = delimeter # Separator for working with .csv file extensions

    def review_to_word_list(self, review: str, remove_stopwords: bool=True) -> list:
        '''Функция преобразует предложение в список слов и возвращает этот список.'''

        # Getting rid of unnecessary data
        review = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", review)

        review_text = BeautifulSoup(review, "lxml").get_text()
        review_text = re.sub("[^а-яА-Я]"," ", review_text)

        words = review_text.lower().split()

        if remove_stopwords:
            stops = stopwords.words("russian")
            words = [w for w in words if w not in stops]
        
        return words
    
    def review_to_sentence(self, review: str, remove_stopwords: bool=False) -> list:
        '''The function converts the text into a list of sentences and returns this list.'''

        raw_sentensces = self.tokenizer.tokenize(review.strip())

        sentences = []

        for raw in raw_sentensces:

            if len(raw_sentensces) > 0:
                sentences.append(self.review_to_word_list(raw, remove_stopwords))

        return sentences
    
    def normolize_words(self, sentenses_list: list) -> list:
        '''The function puts words into the initial form of Russian and returns a two-dimensional list of sentences.'''

        morph = pymorphy2.MorphAnalyzer()
        
        for i in sentenses_list:
            for j in i:

                form_list = morph.normal_forms(j)
                index = i.index(j)
                i[index] = form_list[0]

        return sentenses_list

    def clean_file(self) -> list:
        '''The function clears a file with .csv or .txt extension, 
        transforms it into a dataset for model training, returns a two-dimensional list.'''

        preprocessed_data = []

        # If the .csv file
        if '.csv' in self.file_path:
            clean_sents = []
            
            # Initialising a DataFrame
            data = pd.read_csv(self.file_path, delimiter=self.delimeter).dropna(subset=self.column)
            
            print("Parsing sentences from training set...")

            # Text clearing
            for review in data[self.column]:
                clean_sents += self.review_to_sentence(review, self.tokenizer)

            preprocessed_data = self.normolize_words(clean_sents)

        # If the .txt file
        elif '.txt' in self.file_path:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()

                text = re.sub('\n', ' ', text)
                sents = sent_tokenize(text)

                punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~„“«»†*—/\-‘’'
                clean_sents = []

                # Text clearing
                print("Parsing sentences from training set...")

                for sent in sents:
                    s = [w.lower().strip(punct) for w in sent.split()]
                    clean_sents.append(s)

                preprocessed_data = self.normolize_words(clean_sents)
        
        # If the file extension is unknown
        else:
            print('Datatype is not supported for preprocessing training data.\nPlease, use .csv or .txt file')

        return preprocessed_data


class Semantic_model:
    '''
    The main library class that creates and trains models.
    '''
    def __init__(self, file_path: str, column:str='', workers: int=4, min_count: int=2, window:int=10, sample: float=1e-3):
        '''The function initialises the class.'''
        
        self.file_path = file_path # File path
        self.workers = workers # Number of streams
        self.min_count = min_count # Minimum number of word repetitions to enter the model corpus
        self.window = window # Observation window
        self.sample = sample # Size of downsampling of frequently occurring words
        self.column = column # Column name for working with .csv file extensions

    def make_model(self) -> word2vec.Word2Vec:
        '''The function creates and returns a model of the Word2Vec class, must be saved to a variable.'''

        training_data = Preprocessor(self.file_path, self.column).clean_file()
        model = word2vec.Word2Vec(training_data, workers=self.workers, min_count=self.min_count, window=self.window, sample=self.sample)

        return model
    
    def save_model(self, model: word2vec.Word2Vec, model_path: str='model.model') -> None:
        '''The function saves the model to a non-binary file.'''

        model.save(model_path)

        print(f'Model saved in {model_path}')

    def train_model(self, model_path: str, training_data_path: str, column: str='', epochs: int=5) -> word2vec.Word2Vec:
        '''The function trains a pre-trained user model and returns a new one, must be written to a variable.'''

        # Loading the model
        model = word2vec.Word2Vec.load(model_path)

        # Data preprocessing
        training_data = Preprocessor(training_data_path, column).clean_file()

        # Creating a model dictionary and training on a dataset
        model.build_vocab(training_data, update=True)
        model.train(training_data, total_examples=model.corpus_count, epochs=epochs)

        return model


class Visualiztor:
    '''
    The class visualises the model data into a knowledge graph based on the word fed into it and the closest words to it.
    '''
    def __init__(self, word: str, model_path: str, depth: int=2, topn: int=10):
        '''The function initialises the class.'''

        self.word = word # The word from which the graph is built
        self.model_path = model_path # Word2Vec model path
        self.depth = depth # Depth of graph
        self.topn = topn # Number of nearest words to search for

    def write_nodes_in_dataframe(self):
        '''Write related words as nodes in a graph and cosine similarity as direction to a dataframe.'''

        # Загрузите модель
        model = word2vec.Word2Vec.load(self.model_path)

        # Isolate the key word
        center = [self.word]

        # Create a dictionary for storing nodes and directions
        data = {'subject': [], 'object': [], 'semantic_closeness': []}

        new_list = []

        # Iterate the depth
        for i in range(self.depth):

            # Iterate the number of related words
            loads = len(center)
            for k in range(loads):
                nodes = model.wv.most_similar(center[k], topn=self.topn)

                # Add each node to the dataframe
                for j in nodes:
                    data['subject'].append(center[k])
                    data['object'].append(j[0])
                    data['semantic_closeness'].append(j[1])
                    new_list.append(j[0])

                center = new_list
                new_list = []

        return pd.DataFrame(data)

    def save_graph_img(self, img_path: str='kg1.jpg', options: dict={'node_color': 'yellow',     # Цвет узлов
                                                                    'node_size': 1000,          # Размер узлов
                                                                    'width': 1,                 # Ширина линий связи
                                                                    'arrowstyle': '-|>',        # Стиль стрелки для напрвленного графа
                                                                    'arrowsize': 18,            # Размер стрелки
                                                                    'edge_color':'blue',        # Цвет связи
                                                                    'font_size':20              # Размер шрифта
                                                                    }):
        '''The function creates and saves a graph into an image.'''

        # Initialising a DataFrame
        df = self.write_nodes_in_dataframe()

        # Creating an object to record a graph
        G=nx.from_pandas_edgelist(df,"subject","object", edge_attr=True, create_using=nx.MultiDiGraph())

        # Visualisation into an image
        plt.figure(figsize=(30,30))
        pos = nx.spring_layout(G)
        
        # Visualising the graph into an image and saving it
        nx.draw(G, with_labels=True, pos = pos, **options)
        nx.draw_networkx_edge_labels(G, pos=pos)

        # Save the picture
        plt.savefig(img_path)

        print(f'Image is saved in {img_path}.')
                