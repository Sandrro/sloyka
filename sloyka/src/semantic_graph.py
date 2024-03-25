"""
@class:Semgraph:
The main class of the semantic graph module. It is aimed to build a semantic graph based on the provided data
and parameters.
More convenient to use after extracting data from geocoder.

The Semgraph class has the following methods:

@method:clean_from_dublicates:
A function to clean a DataFrame from duplicates based on specified columns.

@method:clean_from_digits:
Removes digits from the text in the specified column of the input DataFrame.

@method:clean_from_toponims:
Clean the text in the specified text column by removing any words that match the toponims in the name and
toponim columns.

@method:aggregte_data:
Creates a new DataFrame by aggregating the data based on the provided text and toponims columns.
"""
import time
import itertools
from tqdm import tqdm
import re
import torch
import nltk
import pymorphy3
import pandas as pd
import geopandas as gpd
import networkx as nx
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from keybert import KeyBERT
from bertopic import BERTopic

nltk.download('stopwords')

RUS_STOPWORDS = stopwords.words('russian') + ['фото', 'улица', 'дом', 'проспект',
                                              'дорога', 'час', 'год', 'утро',
                                              'здравствуйте', 'ул', 'пр', 'здание',
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
    def clean_from_dublicates(data: pd.DataFrame or gpd.GeoDataFrame,
                              text_column: str,
                              toponim_column: str
                              ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        A function to clean a DataFrame from duplicates based on specified columns.
        
        Args:
            data (pd.DataFrame): The input DataFrame to be cleaned.
            text_column (str): The name of the text column to check for duplicates.
            toponim_column (str): The name of the toponims column.
        
        Returns:
            pd.DataFrame or gpd.GeoDataFrame: A cleaned DataFrame or GeoDataFrame without duplicates based on the
            specified text column.
        """

        uniq_df = data.drop_duplicates(subset=[text_column], keep='first')
        uniq_df = uniq_df.dropna(subset=[text_column, toponim_column])
        uniq_df = uniq_df.reset_index(drop=True)

        return uniq_df

    @staticmethod
    def clean_from_digits(data: pd.DataFrame or gpd.GeoDataFrame,
                          text_column: str
                          ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Removes digits from the text in the specified column of the input DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the text column to clean.

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: The DataFrame with the specified text column cleaned from digits.
        """

        for i in range(len(data)):
            text = str(data[text_column].iloc[i]).lower()
            cleaned_text = ''.join([j for j in text if not j.isdigit()])

            data.at[i, text_column] = cleaned_text

        return data

    @staticmethod
    def clean_from_toponims(data: pd.DataFrame or gpd.GeoDataFrame,
                            text_column: str,
                            name_column: str,
                            toponim_type_column: str
                            ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Clean the text in the specified text column by removing any words that match the toponims in the name
        and toponim columns.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
            text_column (str): The name of the column containing the text to be cleaned.
            name_column (str): The name of the column containing the toponim name (e.g. Nevski, Moika etc).
            toponim_type_column (str): The name of the column containing the toponim type
            (e.g. street, alley, avenue etc).

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: The DataFrame or GeoDataFrame with the cleaned text.
        """

        for i in range(len(data)):

            text = str(data[text_column].iloc[i]).lower()
            word_list = text.split()
            toponims = [str(data[name_column].iloc[i]).lower(), str(data[toponim_type_column].iloc[i]).lower()]

            text = ' '.join([j for j in word_list if j not in toponims])

            data.at[i, text_column] = text

        return data

    @staticmethod
    def clean_from_links(data: pd.DataFrame or gpd.GeoDataFrame,
                         text_column: str
                         ) -> pd.DataFrame or gpd.GeoDataFrame:

        for i in range(len(data)):
            text = str(data[text_column].iloc[i])
            if '[' in text:
                start = text.index('[')
                stop = text.index(']')

                text = text[:start] + text[stop:]

            text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

            data.at[i, text_column] = text

        return data

    @staticmethod
    def fill_empty_toponim(data: pd.DataFrame or gpd.GeoDataFrame,
                           toponim_column: str):

        for i in range(len(data)):
            check = data[toponim_column].iloc[i]
            if check == '':
                data.at[i, toponim_column] = None

        return data

    @staticmethod
    def fill_parents_stack(data: pd.DataFrame or gpd.GeoDataFrame,
                           parents_stack_column: str):

        for i in range(len(data)):
            tmp = data[parents_stack_column].iloc[i]
            if tmp is not None and len(tmp) > 3:
                data.at[i, parents_stack_column] = int(tmp.split()[1])
            else:
                data.at[i, parents_stack_column] = None

        return data

    def extract_keywords(self,
                         data: pd.DataFrame or gpd.GeoDataFrame,
                         text_column: str,
                         text_type_column: str,
                         toponim_column: str,
                         id_column: str,
                         post_id_column: str,
                         parents_stack_column: str,
                         semantic_key_filter: float = 0.6,
                         top_n: int = 1
                         ) -> pd.DataFrame or gpd.GeoDataFrame:

        model = KeyBERT(model=self.model)

        data['words_score'] = None
        data['texts_ids'] = None

        post_top_gdf = data.loc[data[text_type_column] == 'post']
        post_top_gdf = post_top_gdf.dropna(subset=toponim_column)
        post_toponim_list = list(post_top_gdf[id_column])

        comment_top_gdf = data.loc[data[text_type_column] == 'comment']
        comment_top_gdf = comment_top_gdf.dropna(subset=toponim_column)
        comment_toponim_list = list(comment_top_gdf[id_column])

        reply_top_gdf = data.loc[data[text_type_column] == 'reply']
        reply_top_gdf = reply_top_gdf.dropna(subset=toponim_column)
        reply_toponim_list = list(reply_top_gdf[id_column])

        exclude_list = reply_toponim_list + comment_toponim_list

        for i in tqdm(post_toponim_list):

            ids_text_to_extract = list((data[id_column].loc[(data[post_id_column] == i)
                                                           & (~data[id_column].isin(exclude_list))
                                                           & (~data[parents_stack_column].isin(comment_toponim_list))]))

            texts_to_extract = list((data[text_column].loc[(data[post_id_column] == i)
                                                          & (~data[id_column].isin(exclude_list))
                                                          & (~data[parents_stack_column].isin(comment_toponim_list))]))
            ids_text_to_extract.extend(list(data[id_column].loc[data[id_column] == i]))
            texts_to_extract.extend(list(data[text_column].loc[data[id_column] == i]))
            words_to_add = []
            id_to_add = []
            texts_to_add = []

            for j in texts_to_extract:

                extraction = model.extract_keywords(j, top_n=top_n, stop_words=RUS_STOPWORDS)
                if extraction:
                    if extraction[0][1] > semantic_key_filter:
                        word_score = extraction[0]
                        words_to_add.append(word_score)
                        index = texts_to_extract.index(j)
                        id_to_add.append(ids_text_to_extract[index])
                        texts_to_add.append(j)

            if words_to_add:
                index = data.index[data.id == i][0]
                data.at[index, 'words_score'] = words_to_add
                data.at[index, 'texts_ids'] = id_to_add

        for i in tqdm(comment_toponim_list):

            ids_text_to_extract = list(data[id_column].loc[data[parents_stack_column] == i])

            texts_to_extract = list(data[text_column].loc[data[parents_stack_column] == i])

            ids_text_to_extract.extend(list(data[id_column].loc[data[id_column] == i]))
            texts_to_extract.extend(data[text_column].loc[data[id_column] == i])
            words_to_add = []
            id_to_add = []
            texts_to_add = []

            for j in texts_to_extract:

                extraction = model.extract_keywords(j, top_n=top_n, stop_words=RUS_STOPWORDS)
                if extraction:
                    if extraction[0][1] > semantic_key_filter:
                        word_score = extraction[0]
                        words_to_add.append(word_score)
                        index = texts_to_extract.index(j)
                        id_to_add.append(ids_text_to_extract[index])
                        texts_to_add.append(j)

            if words_to_add:
                index = data.index[data.id == i][0]
                data.at[index, 'words_score'] = words_to_add
                data.at[index, 'texts_ids'] = id_to_add

        for i in tqdm(reply_toponim_list):

            id_text_to_extract = data[id_column].loc[data[id_column] == i]

            text_to_extract = data[text_column].loc[data[id_column] == i]

            words_to_add = []
            id_to_add = []
            texts_to_add = []

            for j in text_to_extract:

                extraction = model.extract_keywords(j, top_n=top_n, stop_words=RUS_STOPWORDS)
                if extraction:
                    if extraction[0][1] > semantic_key_filter:
                        word_score = extraction[0]
                        words_to_add.append(word_score)
                        texts_to_add.append(j)

            if words_to_add:
                index = data.index[data.id == i][0]
                data.at[index, 'words_score'] = words_to_add
                data.at[index, 'texts_ids'] = id_text_to_extract

        df_to_graph = data.dropna(subset='words_score')

        return df_to_graph


    def convert_df_to_edge_df(self,
                              data: pd.DataFrame or gpd.GeoDataFrame,
                              toponim_column: str,
                              word_and_score_column: str = 'words_score'
                              ) -> pd.DataFrame or gpd.GeoDataFrame:

        morph = pymorphy3.MorphAnalyzer()

        edge_list = []

        for i in data[toponim_column]:
            current_df = data.loc[data[toponim_column] == i]
            for j in range(len(current_df)):
                toponim = current_df[toponim_column].iloc[j]
                word_nodes = current_df[word_and_score_column].iloc[j]

                for k in word_nodes:
                    p = morph.parse(k[0])[0]
                    if p.tag.POS in ['NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN']:
                        edge_list.append([toponim, p.normal_form, k[1]])

        edge_df = pd.DataFrame(edge_list, columns=['FROM', 'TO', 'SCORE'])

        return edge_df

    def get_semantic_closeness(self,
                               data: pd.DataFrame or gpd.GeoDataFrame,
                               column: str,
                               similarity_filter: float = 0.75
                               ) -> pd.DataFrame or gpd.GeoDataFrame:
        """
        Calculate the semantic closeness between unique words in the specified column of the input DataFrame.

        Args:
            data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
            column (str): The column in the DataFrame to calculate semantic closeness for.
            similarity_filter (float = 0.75): The score of cosinus semantic proximity, from which and upper the edge
            will be generated.

        Returns:
            pd.DataFrame or gpd.GeoDataFrame: A DataFrame or GeoDataFrame containing the pairs of words with their
            similarity scores.
        """

        unic_words = tuple(set(data[column]))
        words_tokens = tuple(
            [self.tokenizer.encode(i, add_special_tokens=False, return_tensors='pt').to(self.device) for i in
             unic_words])
        potential_new_nodes_embendings = tuple(
            [[unic_words[i], self.model(words_tokens[i]).last_hidden_state.mean(dim=1)] for i in
             range(len(unic_words))])
        new_nodes = []

        combinations = list(itertools.combinations(potential_new_nodes_embendings, 2))

        print('Calculating semantic closeness')
        for word1, word2 in tqdm(combinations):

            similarity = float(torch.nn.functional.cosine_similarity(word1[1], word2[1]))

            if similarity >= similarity_filter:
                new_nodes.append([word1[0], word2[0], similarity])

            time.sleep(0.001)

        result_df = pd.DataFrame(new_nodes, columns=['FROM', 'TO', 'SIMILARITY_SCORE'])

        return result_df

    @staticmethod
    def get_tag(nodes: list,
                       toponims: list
                       ) -> dict:
        """
        Get attributes of part of speech for the given nodes, with the option to specify toponims.
        
        Args:
            nodes (list): list of strings representing the nodes
            toponims (list): list of strings representing the toponims

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

    @staticmethod
    def get_coordinates(G: nx.classes.graph.Graph,
                        geocoded_data: gpd.GeoDataFrame,
                        toponim_column: str,
                        location_column: str,
                        geometry_column: str
                        ) -> nx.classes.graph.Graph:
        """
        Get and write coordinates from geometry column in gpd.GeoDataFrame.

        Args:
            G (nx.classes.graph.Graph): Prebuild input graph.
            geocoded_data (gpd.GeoDataFrame): Data containing toponim, location and geometry of toponim.
            toponim_column (str): The name of the column containing the toponim data.
            location_column (str): The name of the column containing the location data.
            geometry_column (str): The name of the column containing the geometry data.

        Returns:
            nx.classes.graph.Graph: Graph with toponim nodes ('tag'=='TOPONIM') containing information
            about address and geometry ('Location','Lon','Lat' as node attributes)
        """
        toponims_list = [i for i in G.nodes if G.nodes[i].get('tag') == 'TOPONIM']
        all_toponims_list = list(geocoded_data[toponim_column])

        for i in toponims_list:
            if i in all_toponims_list:
                index = all_toponims_list.index(i)
                G.nodes[i]['Location'] = str(geocoded_data[location_column].iloc[all_toponims_list.index(i)])

        for i in toponims_list:
            if i in all_toponims_list:
                cord = geocoded_data[geometry_column].iloc[all_toponims_list.index(i)]
                if cord is not None:
                    G.nodes[i]['Lat'] = cord.x
                    G.nodes[i]['Lon'] = cord.y

        return G

    @staticmethod
    def get_text_ids(G: nx.classes.graph.Graph,
                     filtered_data: pd.DataFrame or gpd.GeoDataFrame,
                     toponim_column: str,
                     text_id_column: str = 'texts_ids'
                     ) -> pd.DataFrame or gpd.GeoDataFrame:

        toponims_list = [i for i in G.nodes if G.nodes[i]['tag'] != 'TOPONIM']

        for i in toponims_list:
            df_id_text = filtered_data.loc[filtered_data[toponim_column] == i]

            ids = []
            for j in range(len(df_id_text)):
                ids.extend(filtered_data[text_id_column].iloc[j])

            ids = [str(int(j)) for j in ids]
            G.nodes[i][text_id_column] = ','.join(ids)

        return G

    # @staticmethod
    # def graph_to_key_words_

    def run(self,
                             data: pd.DataFrame or gpd.GeoDataFrame,
                             id_column: str,
                             text_column: str,
                             text_type_column: str,
                             toponim_column: str,
                             toponim_name_column: str,
                             toponim_type_column: str,
                             post_id_column: str,
                             parents_stack_column: str,
                             location_column: str or None = None,
                             geometry_column: str or None = None,
                             key_score_filter: float = 0.6,
                             semantic_score_filter: float = 0.75,
                             top_n: int = 1
                             ) -> nx.classes.graph.Graph:

        """
        Builds a semantic graph based on the provided data and parameters.

        Args::
            data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame or GeoDataFrame containing the data.
            text_column (str): The name of the column containing the text data.
            toponim_column (str): The name of the column containing the toponim data.
            toponim_name_column (str): The name of the column containing the toponim name data in text.
            toponim_type_column (str): The name of the column containing the toponim type data.
            location_column (str): The name of the column containing the toponims address str.
            Use only with GeoDataFrame
            geometry_column (str): The name of the column containing the toponims geometry as a point.
            Use only with GeoDataFrame
            key_score_filter (float): The threshold for key-extracting score filtering.
            semantic_score_filter (float, optional): The threshold for semantic score filtering.
            top_n (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            nx.classes.graph.Graph: The semantic graph constructed from the input data.
        """

        data = self.clean_from_dublicates(data,
                                          text_column,
                                          toponim_column)

        data = self.clean_from_digits(data,
                                      text_column)

        data = self.clean_from_toponims(data,
                                        text_column,
                                        toponim_name_column,
                                        toponim_type_column)

        data = self.clean_from_links(data,
                                     text_column)

        data = self.fill_empty_toponim(data,
                                       toponim_column)

        data = self.fill_parents_stack(data,
                                       parents_stack_column)

        df = self.extract_keywords(data,
                                   text_column,
                                   text_type_column,
                                   toponim_column,
                                   id_column,
                                   post_id_column,
                                   parents_stack_column,
                                   key_score_filter,
                                   top_n)

        preprocessed_df = self.convert_df_to_edge_df(df,
                                                     toponim_column)

        words_df = self.get_semantic_closeness(preprocessed_df,
                                               'TO',
                                               semantic_score_filter)

        graph_df = pd.concat([preprocessed_df, words_df],
                             ignore_index=True)

        G = nx.from_pandas_edgelist(graph_df,
                                    source='FROM',
                                    target='TO',
                                    edge_attr='SIMILARITY_SCORE')

        nodes = list(G.nodes())
        attributes = self.get_tag(nodes, set(data[toponim_column]))

        nx.set_node_attributes(G, attributes, 'tag')

        if type(data) is gpd.GeoDataFrame:
            G = self.get_coordinates(G=G,
                                     geocoded_data=data,
                                     toponim_column=toponim_column,
                                     location_column=location_column,
                                     geometry_column=geometry_column)

        G = self.get_text_ids(G=G,
                              filtered_data=df,
                              toponim_column=toponim_column)

        return G

# debugging

if __name__ == '__main__':

    file = open("C:\\Users\\thebe\\Downloads\\Telegram Desktop\\df_vyborg_geocoded.geojson", encoding='utf-8')
    test_gdf = gpd.read_file(file)

    sm = Semgraph(device='cuda')

    G = sm.run(test_gdf,
               id_column='id',
               text_column='text',
               text_type_column='type',
               toponim_column='only_full_street_name',
               toponim_name_column='initial_street',
               toponim_type_column='Toponims',
               post_id_column='post_id',
               parents_stack_column='parents_stack',
               location_column='Location',
               geometry_column='geometry')

    nx.write_graphml(G, 'vyborg_graph.graphml', encoding='utf-8')

