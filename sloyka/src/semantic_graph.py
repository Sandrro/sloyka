import pandas as pd
import pymorphy3
from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import nltk 
from nltk.corpus import stopwords 
from pyvis.network import Network
import networkx as nx
import time
from tqdm import tqdm
from pyvis.network import Network
import networkx as nx
import itertools
import torch

nltk.download('stopwords')


class Semgraph:

    def __init__(self,
                 bert_name: str = 'DeepPavlov/rubert-base-cased',
                 language: str = 'russian',
                 device: str = 'cpu') -> None:
        """
    	Initialize the class with the specified BERT model name, language, and device for inference.

    	:param bert_name: the name of the BERT model to use (default is 'DeepPavlov/rubert-base-cased')
    	:param language: the language of the BERT model (default is 'russian')
    	:param device: the device to use for inference (default is 'cpu')
    	:return: None
    	"""

        self.language = language
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.model = BertModel.from_pretrained(bert_name.to(device))

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
                            toponim_type_column
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

    @staticmethod
    def aggregte_data(data: pd.DataFrame,
                              text_colum: str,
                              toponims_column: str,
                              ) -> pd.DataFrame:
        """
        Creates a new DataFrame by aggregating the data based on the provided text and toponims columns.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing the data to be aggregated.
            text_colum (str): The name of the column containing the text data.
            toponims_column (str): The name of the column containing the toponims data (e.g. Nevski Prospect, Moika enbankment, etc).
            
        Returns:
            pd.DataFrame: A new DataFrame with aggregated data based on the provided columns.
        """

        toponims = list(set(data[toponims_column]))
        new_df_rows = []
        for i in toponims:
            tmp_df = data.loc[data[text_colum] == i].reset_index(drop=True)
            text = str(tmp_df[toponims_column].iloc[0])
            for j in range(1, len(tmp_df)):

                text = text + ' ' + str(tmp_df[toponims_column].iloc[j])

            new_df_rows.append([text, i])

        new_data = pd.DataFrame(new_df_rows, columns = [toponims_column, text_colum])
        
        return new_data

    def extract_keywords(self,
                         data: pd.DataFrame,
                         text_column: str,
                         toponim_column: str,
                         toponim_name_column: str,
                         toponim_type_column: str,
                         semantic_score_filter: float,
                         top_n: int=5,) -> pd.DataFrame:
        """
        Extracts keywords from the given data using KeyBERT, cleans the data from digits and toponims, aggregates the data, and extracts keywords for each toponim with a semantic score filter. Returns a DataFrame containing the extracted keywords, their associated toponim, and their semantic score.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing the text and toponim data.
            text_column (str): The name of the column containing the text data.
            toponim_column (str): The name of the column containing the toponim data.
            toponim_name_column (str): The name of the column containing the toponim names.
            toponim_type_column (str): The name of the column containing the toponim types.
            semantic_score_filter (float): The minimum semantic score required for a keyword to be included.
            top_n (int, optional): The number of top keywords to extract for each toponim. Defaults to 5.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted keywords, their associated toponim, and their semantic similarity score.
        """

        keybert_model = KeyBERT(model=self.model)
        morph = pymorphy3.MorphAnalyzer()
        nodes = []

        data = Semgraph.clean_from_digits(data, text_column=text_column)
        data = Semgraph.clean_from_toponims(data, 
                                            text_column=text_column,
                                            name_column=toponim_name_column,
                                            toponim_type_column=toponim_type_column)

        consolidated_df = Semgraph.aggregte_data(data,
                                                 text_colum=text_column,
                                                 toponims_column= toponim_column)

        for i in tqdm(range(len(consolidated_df))):

            toponim = consolidated_df[toponim_column].iloc[i]
            text = consolidated_df[text_column].iloc[i]

            keywords_list = keybert_model.extract_keywords(text,
                                                           top_n=top_n,
                                                           stop_words=stopwords.words(self.language))
            # new_list = [j for j in keywords_list]
            for j in (keywords_list):
                if j[1] >= semantic_score_filter:
                    p = morph.parse(j[0])[0]
                    nodes.append([toponim, p.normal_form, j[1]])

            time.sleep(0.01)

        new_df = pd.DataFrame(nodes, columns=['FROM', 'TO', 'SIMILARITY_SCORE'])

        return new_df
    
    def get_semantic_closeness(self, data: pd.DataFrame,
                               column: str,
                               similaryty_filter: float = 0.8) -> pd.DataFrame:
        """
    	Calculate the semantic closeness between unique words in the specified column of the input DataFrame.
    	
    	Args:
    	    data (pd.DataFrame): The input DataFrame.
    	    column (str): The column in the DataFrame to calculate semantic closeness for.
    	
    	Returns:
    	    pd.DataFrame: A DataFrame containing the pairs of words with their similarity scores.
    	"""

        unic_words = tuple(set(data[column]))
        words_tokens = tuple([self.tokenizer.encode(i, add_special_tokens=False, return_tensors='pt').to('cuda') for i in unic_words])
        potential_new_nodes_embendings = tuple([[unic_words[i], self.model(words_tokens[i]).last_hidden_state.mean(dim=1)] for i in range(len(unic_words))])
        new_nodes = []

        combinations = list(itertools.combinations(potential_new_nodes_embendings, 2))

        for word1, word2 in tqdm(combinations):

            similarity = torch.nn.functional.cosine_similarity(word1[1], word2[1])

            if similarity >= similaryty_filter:

                new_nodes.append([word1[0], word2[0], similarity])
            


            time.sleep(0.01)


        result_df = pd.DataFrame(new_nodes, columns = ['FROM','TO', 'SIMILARITY_SCORE'])

        return result_df
    
    def build_semantic_graph(self,
                             data: pd.DataFrame,
                             ) -> nx.classes.graph.Graph: