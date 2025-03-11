import time

import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
import pymorphy2

from soika.src.utils.constants import STOPWORDS, TAG_ROUTER
    

def extract_keywords(
    self,
    data: pd.DataFrame or gpd.GeoDataFrame,
    text_column: str,
    text_type_column: str,
    toponym_column: str,
    id_column: str,
    post_id_column: str,
    parents_stack_column: str,
    semantic_key_filter: float = 0.6,
    top_n: int = 1,
) -> pd.DataFrame or gpd.GeoDataFrame:
    """
    Извлекает ключевые слова из данных с учётом заданных критериев.

    Args:
        data (pd.DataFrame или gpd.GeoDataFrame): входные данные.
        text_column (str): столбец с текстом.
        text_type_column (str): столбец, указывающий тип текста (например, пост, комментарий, ответ).
        toponym_column (str): столбец с топонимами.
        id_column (str): столбец с уникальными идентификаторами текста.
        post_id_column (str): столбец с идентификаторами постов (для комментариев и ответов).
        parents_stack_column (str): столбец с информацией о родительско-дочерних связях.
        semantic_key_filter (float, optional): порог семантического фильтра. По умолчанию 0.6.
        top_n (int, optional): число извлекаемых ключевых слов. По умолчанию 1.

    Returns:
        pd.DataFrame или gpd.GeoDataFrame: обработанные данные с извлечёнными ключевыми словами, счетчиками топонимов и словами.
    """
    nltk.download("stopwords")
    RUS_STOPWORDS = stopwords.words(self.language) + STOPWORDS

    morph = pymorphy2.MorphAnalyzer()

    data["words_score"] = None
    data["texts_ids"] = None

    toponym_dict = {}
    word_dict = {}

    chains = ["post", "comment", "reply"]

    for chain in chains:
        chain_gdf = data.loc[data[text_type_column] == chain]
        chain_gdf = chain_gdf.dropna(subset=toponym_column)
        chain_toponym_list = list(chain_gdf[id_column])

        exclude_list = []

        print(f"Extracting keywords from {chain} chains...")
        time.sleep(1)

        for i in tqdm(chain_toponym_list):
            toponym = data[toponym_column].loc[data[id_column] == i].iloc[0]

            ids_text_to_extract = list(
                data[id_column].loc[
                    (data[post_id_column] == i)
                    & (~data[id_column].isin(exclude_list))
                    & (~data[parents_stack_column].isin(chain_toponym_list))
                ]
            )

            texts_to_extract = list(
                data[text_column].loc[
                    (data[post_id_column] == i)
                    & (~data[id_column].isin(exclude_list))
                    & (~data[parents_stack_column].isin(chain_toponym_list))
                ]
            )

            ids_text_to_extract.extend(
                list(data[id_column].loc[data[id_column] == i])
            )
            texts_to_extract.extend(
                list(data[text_column].loc[data[id_column] == i])
            )
            words_to_add = []
            id_to_add = []
            texts_to_add = []

            for j, text in zip(ids_text_to_extract, texts_to_extract):
                extraction = KeyBERT().extract_keywords(
                    docs=text, top_n=top_n, stop_words=RUS_STOPWORDS
                )
                if extraction:
                    score = extraction[0][1]
                    if score > semantic_key_filter:
                        word_score = extraction[0]
                        p = morph.parse(word_score[0])[0]
                        if p.tag.POS in TAG_ROUTER.keys():
                            word = p.normal_form
                            tag = p.tag.POS

                            word_info = (word, score, tag)

                            words_to_add.append(word_info)
                            id_to_add.append(j)
                            texts_to_add.append(text)

                            word_dict[word] = word_dict.get(word, 0) + 1

            if words_to_add:
                toponym_dict[toponym] = toponym_dict.get(toponym, 0) + 1

                index = data.index[data.index == i][0]
                data.at[index, "words_score"] = words_to_add
                data.at[index, "texts_ids"] = id_to_add

        exclude_list += chain_toponym_list

    df_to_graph = data.dropna(subset="words_score")

    return [df_to_graph, toponym_dict, word_dict]
