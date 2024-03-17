"""
@class:Semgraph:
The main class of the semantic graph module. It is aimed to build a semantic graph based on the provided data and parameters.
More convenient to use after extracting data from geocoder.

The Semgraph class has the following methods:

@method:clean_from_dublicates:
A function to clean a DataFrame from duplicates based on specified columns.

@method:clean_from_digits:
Removes digits from the text in the specified column of the input DataFrame.

@method:clean_from_toponims:
Clean the text in the specified text column by removing any words that match the toponims in the name and toponim columns.

@method:aggregte_data:
Creates a new DataFrame by aggregating the data based on the provided text and toponims columns.
"""
import time
import itertools
from tqdm import tqdm
import torch
import nltk
import pymorphy3
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from keybert import KeyBERT
from bertopic import BERTopic


nltk.download('stopwords')


RUS_STOPWORDS = stopwords.words('russian') + ['фото', 'улица', 'дом', 'проспект',
                                              'дорога', 'час', 'год', 'утро', 
                                              'здравствуйте', 'ул','пр', 'здание',
                                              'город', 'аноним', 'утро', 'день',
                                              'вечер']


class Semgraph:
    """
    This is the main class of semantic graph module. 
    It is aimed to build a semantic graph based on the provided data and parameters.
    More convinient to use after extracting data from geocoder.

    Param:
    bert_name: the name of the BERT model to use (default is 'DeepPavlov/rubert-base-cased')
    language: the language of the BERT model (default is 'russian')
    device: the device to use for inference (default is 'cpu')
    """
    def __init__(self,
                 bert_name: str = 'DeepPavlov/rubert-base-cased',
                 language: str = 'russian',
                 device: str = 'cpu'
                 ) -> None:

        self.language = language
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.model_name = bert_name
        self.model = BertModel.from_pretrained(bert_name).to(device)

    @staticmethod
    def clean_from_dublicates(data: pd.DataFrame,
                              text_column: str,
                              toponim_column: str
                              ) -> pd.DataFrame:
        """
        A function to clean a DataFrame from duplicates based on specified columns.
        
        Args:
            data (pd.DataFrame): The input DataFrame to be cleaned.
            text_column (str): The name of the text column to check for duplicates.
            toponim_column (str): The name of the toponims column.
        
        Returns:
            pd.DataFrame: A cleaned DataFrame without duplicates based on the specified text column.
        """

        uniq_df = data.drop_duplicates(subset=[text_column], keep='first')
        uniq_df = uniq_df.dropna(subset=[text_column, toponim_column])
        uniq_df = uniq_df.reset_index(drop=True)

        return uniq_df

    @staticmethod
    def clean_from_digits(data: pd.DataFrame,
                          text_column: str
                          ) -> pd.DataFrame:
        """
    	Removes digits from the text in the specified column of the input DataFrame.
    	
    	Args:
    	    data (pd.DataFrame): The input DataFrame.
    	    text_column (str): The name of the text column to clean.
    	
    	Returns:
    	    pd.DataFrame: The DataFrame with the specified text column cleaned from digits.
    	"""

        for i in range(len(data)):
            text = str(data[text_column].iloc[i]).lower()
            cleaned_text = ''.join([j for j in text if not j.isdigit()])

            data.at[i, text_column] = cleaned_text

        return data

    @staticmethod
    def clean_from_toponims(data: pd.DataFrame,
                            text_column: str,
                            name_column: str,
                            toponim_type_column: str
                            ) -> pd.DataFrame:
        """
    	Clean the text in the specified text column by removing any words that match the toponims in the name and toponim columns. 
    	
    	Args:
    	    data (pd.DataFrame): The input DataFrame.
    	    text_column (str): The name of the column containing the text to be cleaned.
    	    name_column (str): The name of the column containing the toponim name (e.g. Nevski, Moika etc).
    	    toponim_type_column: The name of the column containing the toponim type (e.g. street, alley, avenue etc).
    	
    	Returns:
    	    pd.DataFrame: The DataFrame with the cleaned text.
    	"""

        for i in range(len(data)):

            text = str(data[text_column].iloc[i]).lower()
            word_list = text.split()
            toponims = [str(data[name_column].iloc[i]).lower(), str(data[toponim_type_column].iloc[i]).lower()]

            text = ' '.join([j for j in word_list if j not in toponims])

            data.at[i, text_column] = text

        return data

    def aggregate_data(self,
                      data: pd.DataFrame,
                      text_column: str,
                      toponims_column: str
                      ) -> pd.DataFrame:
        """
    	Aggregate data based on toponims and cluster it using BERTopic model.
        
    	Args:
            data: pd.DataFrame, the input data
            text_column: str, the name of the column containing the text data
            toponims_column: str, the name of the column containing the toponims
        
        Returns:
            pd.DataFrame, the aggregated and clustered data
    	"""

        toponims = list(set(data[toponims_column]))
        new_df_rows = []

        model = BERTopic(embedding_model=self.model_name)

        print('Clasterizing data...')
        for i in tqdm(toponims):

            if i is not None:
                tmp_df = data.loc[data[toponims_column] == i].reset_index(drop=True)
                if len(tmp_df) >= 10:
                    topics, _ = model.fit_transform(tmp_df[text_column])
                    tmp_df['topic'] = topics

                    topic_names = list(set(topics))

                    for j in topic_names:

                        clustered_df = tmp_df.loc[tmp_df['topic'] == j].reset_index(drop=True)
                        if len(clustered_df) > 0:
                            text = clustered_df[text_column].iloc[0]

                            for k in range(1, len(clustered_df)):

                                text = text + ' ' + str(clustered_df[text_column].iloc[k])

                            new_df_rows.append([text, i, j])

                else:

                    text = tmp_df[text_column].iloc[0]
                    for j in range(1, len(tmp_df)):

                        text = text + ' ' + str(tmp_df[text_column].iloc[j])

                    new_df_rows.append([text, i, None])

                new_data = pd.DataFrame(new_df_rows, columns = [text_column, toponims_column, 'cluster'])
            
            time.sleep(0.01)

        return new_data

    def extract_keywords(self,
                         data: pd.DataFrame,
                         text_column: str,
                         toponim_column: str,
                         toponim_name_column: str,
                         toponim_type_column: str,
                         semantic_key_filter: float=0.4,
                         top_n: int=5
                         ) -> pd.DataFrame:
        """
        Extracts keywords from the given data using KeyBERT, cleans the data from digits and toponims, aggregates the data, and extracts keywords for each toponim with a semantic score filter. Returns a DataFrame containing the extracted keywords, their associated toponim, and their semantic score.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing the text and toponim data.
            text_column (str): The name of the column containing the text data.
            toponim_column (str): The name of the column containing the toponim data.
            toponim_name_column (str): The name of the column containing the toponim names.
            toponim_type_column (str): The name of the column containing the toponim types.
            semantic_key_filter (float): The minimum semantic score required for a keyword to be included.
            top_n (int, optional): The number of top keywords to extract for each toponim. Defaults to 5.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted keywords, their associated toponim, and their semantic similarity score.
        """

        keybert_model = KeyBERT(model=self.model)
        morph = pymorphy3.MorphAnalyzer()
        nodes = []

        data = Semgraph.clean_from_dublicates(data,
                                              text_column=text_column,
                                              toponim_column=toponim_column)
        
        data = Semgraph.clean_from_digits(data,
                                          text_column=text_column)
        
        data = Semgraph.clean_from_toponims(data,
                                            text_column=text_column,
                                            name_column=toponim_name_column,
                                            toponim_type_column=toponim_type_column)

        consolidated_df = self.aggregate_data(data,
                                                 text_column=text_column,
                                                 toponims_column=toponim_column)

        print('Extracting keywords')
        for i in tqdm(range(len(consolidated_df))):

            toponim = consolidated_df[toponim_column].iloc[i]
            text = consolidated_df[text_column].iloc[i]

            if self.language == 'russian':
                full_stopwords = RUS_STOPWORDS

            else:
                full_stopwords = stopwords.words(self.language)

            keywords_list = keybert_model.extract_keywords(text,
                                                           top_n=top_n,
                                                           stop_words=full_stopwords)

            for j in (keywords_list):
                if j[1] >= semantic_key_filter:
                    p = morph.parse(j[0])[0]
                    if p.tag.POS in ['NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN']:
                        nodes.append([toponim, p.normal_form, j[1]])

            time.sleep(0.01)


        new_df = pd.DataFrame(nodes, columns=['FROM', 'TO', 'SIMILARITY_SCORE'])

        return new_df
    
    def get_semantic_closeness(self, data: pd.DataFrame,
                               column: str,
                               similaryty_filter: float = 0.6
                               ) -> pd.DataFrame:
        """
    	Calculate the semantic closeness between unique words in the specified column of the input DataFrame.
    	
    	Args:
    	    data (pd.DataFrame): The input DataFrame.
    	    column (str): The column in the DataFrame to calculate semantic closeness for.
    	
    	Returns:
    	    pd.DataFrame: A DataFrame containing the pairs of words with their similarity scores.
    	"""

        unic_words = tuple(set(data[column]))
        words_tokens = tuple([self.tokenizer.encode(i, add_special_tokens=False, return_tensors='pt').to(self.device) for i in unic_words])
        potential_new_nodes_embendings = tuple([[unic_words[i], self.model(words_tokens[i]).last_hidden_state.mean(dim=1)] for i in range(len(unic_words))])
        new_nodes = []

        combinations = list(itertools.combinations(potential_new_nodes_embendings, 2))

        print('Calculating semantic closeness')
        for word1, word2 in tqdm(combinations):

            similarity = float(torch.nn.functional.cosine_similarity(word1[1], word2[1]))

            if similarity >= similaryty_filter:

                new_nodes.append([word1[0], word2[0], similarity])
            


            time.sleep(0.01)


        result_df = pd.DataFrame(new_nodes, columns = ['FROM','TO', 'SIMILARITY_SCORE'])

        return result_df
    
    @staticmethod
    def get_attributes(nodes: list,
                       toponims: list
                       ) -> dict:
        """
        Get attributes of part of speech for the given nodes, with the option to specify toponims.
        
        Args:
            nodes: list of strings representing the nodes
            toponims: list of strings representing the toponims

        Returns: 
            dict: dictionary containing attributes for the nodes
        """
                
        morph = pymorphy3.MorphAnalyzer()
        attrs = {}

        for i in nodes:
            if i not in toponims:
                attrs[i] = str(morph.parse(i)[0].tag.POS)
            else:
                attrs[i] = 'TOPONIM'

        return attrs
    
    def build_semantic_graph(self,
                             data: pd.DataFrame,
                             text_column: str,
                             toponim_column: str,
                             toponim_name_column: str,
                             toponim_type_column: str,
                             key_score_filter: float = 0.6,
                             semantic_score_filter: float = 0.4,
                             top_n: int=5
                             ) -> nx.classes.graph.Graph:
        
        """
        Builds a semantic graph based on the provided data and parameters.

        Args::
            data (pd.DataFrame): The input dataframe containing the data.
            text_column (str): The name of the column containing the text data.
            toponim_column (str): The name of the column containing the toponim data.
            toponim_name_column (str): The name of the column containing the toponim name data in text.
            toponim_type_column (str): The name of the column containing the toponim type data.
            key_score_filter (float): The threshold for key-extracting score filtering.
            semantic_score_filter (float): The threshold for semantic score filtering.
            top_n (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            nx.classes.graph.Graph: The semantic graph constructed from the input data.
        """

        df = self.extract_keywords(data,
                                   text_column,
                                   toponim_column,
                                   toponim_name_column,
                                   toponim_type_column,
                                   key_score_filter,
                                   top_n)
        
        words_df = self.get_semantic_closeness(df, 'TO', semantic_score_filter)

        graph_df = pd.concat([df, words_df], ignore_index=True)

        G = nx.from_pandas_edgelist(graph_df,
                                    source='FROM',
                                    target='TO',
                                    edge_attr='SIMILARITY_SCORE')
        
        nodes = list(G.nodes())
        attributes = self.get_attributes(nodes, set(data[toponim_column]))

        nx.set_node_attributes(G, attributes, 'tag')

        return G
