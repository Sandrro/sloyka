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

    def __init__(self, 
                 training_df: pd.DataFrame,
                 text_column: str,
                 street_text_colum: str=None,
                 street_clean_column: str=None) -> None:
        '''The function initialises the class.'''

        self.training_df = training_df                                      # Trining dataframe
        self.tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')  # Receipt of the tokeniser
        self.text_column = text_column                                      # text_ name for working with .csv file extensions
        self.street_text_colum = street_text_colum                          # Location name in original form in corpus
        self.street_clean_column = street_clean_column                      # Location name in clean form

    def review_to_word_list(self, review: str, remove_stopwords: bool=True) -> list:
        '''The function converts a sentence into a list of words and returns that list.'''

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
    
    def normolize_words(self, sentenses_list: list, street_names: pd.DataFrame=None) -> list:
        '''The function puts words into the initial form of Russian and returns a two-dimensional list of sentences.'''

        morph = pymorphy2.MorphAnalyzer()
        if  not street_names.empty:
            for i in range(len(sentenses_list)):
                for j in range(len(sentenses_list[i])):
                    if sentenses_list[i][j] in street_names[self.street_text_colum].values:
                        sentenses_list[i][j] = street_names.loc[street_names[self.street_text_colum] == sentenses_list[i][j], self.street_clean_column].values[0]

        for i in sentenses_list:
            for j in i:
                # if j not in street_names['Location'].values:
                form_list = morph.normal_forms(j)
                index = i.index(j)
                i[index] = form_list[0]

        return sentenses_list

    def clean_file(self) -> list:
        '''The function clears a file with .csv or .txt extension, 
        transforms it into a dataset for model training, returns a two-dimensional list.'''

        preprocessed_data = []

        clean_sents = []
            
        # Initialising a DataFrame
        data = self.training_df.dropna(subset=self.text_column)
            
        print("Parsing sentences from training set...")

        # Text clearing
        for review in data[self.text_column]:
            clean_sents += self.review_to_sentence(review, self.tokenizer)

        # street names cleaning
        if self.street_text_colum is not None and self.street_clean_column is not None:
            street_names = data[[self.street_text_colum, self.street_clean_column]].copy()
            preprocessed_data = self.normolize_words(clean_sents, street_names)

        else:
            preprocessed_data = self.normolize_words(clean_sents)

        return preprocessed_data


class Semanticmodel:
    '''
    The main library class that creates and trains semantic models.
    '''
    def __init__(self, training_df: str, 
                 text_column:str,
                 text_street_column: str=None,
                 street_clean_column: str=None, 
                 workers: int=4, 
                 min_count: int=1, 
                 window:int=10, 
                 sample: float=1e-3):
        '''The function initialises the class.'''
        
        self.training_df = training_df                  # training dataframe
        self.text_streat_column = text_street_column    # Location name in original form in corpus
        self.street_clean_column = street_clean_column  # Location name in clean form
        self.workers = workers                          # Number of streams
        self.min_count = min_count                      # Minimum number of word repetitions to enter the model corpus
        self.window = window                            # Observation window
        self.sample = sample                            # Size of downsampling of frequently occurring words
        self.text_column = text_column                  # text column og pandas df

    def make_model(self) -> word2vec.Word2Vec:
        '''The function creates and returns a model of the Word2Vec class, must be saved to a variable.'''

        training_data = Preprocessor(self.training_df, 
                                     self.text_column, 
                                     self.text_streat_column, 
                                     self.street_clean_column).clean_file()
        
        model = word2vec.Word2Vec(training_data, 
                                  workers=self.workers, 
                                  min_count=self.min_count, 
                                  window=self.window, 
                                  sample=self.sample)

        return model
     
    def save_model(self, model: word2vec.Word2Vec, model_path: str='model.model') -> None:
        '''The function saves the model to a non-binary file.'''

        model.save(model_path)

        print(f'Model saved in {model_path}')

    def train_model(self, model_path: str, epochs: int=5) -> word2vec.Word2Vec:
        '''The function trains a pre-trained user model and returns a new one, must be written to a variable.'''

        # Loading the model
        model = word2vec.Word2Vec.load(model_path)

        # Data preprocessing
        training_data = Preprocessor(self.training_df, self.text_column, self.text_streat_column, self.street_clean_column).clean_file()

        # Creating a model dictionary and training on a dataset
        model.build_vocab(training_data, update=True)
        model.train(training_data, total_examples=model.corpus_count, epochs=epochs)

        return model


class Visualizator:
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

    def save_graph_img(self, img_path: str='kg1.jpg') -> None:
        '''The function creates and saves a graph into an image.'''

        options = {'node_color': 'yellow',  # Цвет узлов
                   'node_size': 1000,       # Размер узлов
                   'width': 1,              # Ширина линий связи
                   'arrowstyle': '-|>',     # Стиль стрелки для напрвленного графа
                   'arrowsize': 18,         # Размер стрелки
                   'edge_color':'blue',     # Цвет связи
                   'font_size':20}          # Размер шрифта

        # Initialising a DataFrame
        df = self.write_nodes_in_dataframe()

        # Creating an object to record a graph
        g = nx.from_pandas_edgelist(df,"subject","object", edge_attr=True, create_using=nx.MultiDiGraph())

        # Visualisation into an image
        plt.figure(figsize=(30,30))
        pos = nx.spring_layout(g)
        
        # Visualising the graph into an image and saving it
        nx.draw(g, with_labels=True, pos = pos, **options)
        nx.draw_networkx_edge_labels(g, pos=pos)

        # Save the picture
        plt.savefig(img_path)

        print(f'Image is saved in {img_path}.')
                