import re

import pandas as pd
import geopandas as gpd


@staticmethod
def clean_from_dublicates(data: pd.DataFrame or gpd.GeoDataFrame, id_column: str) -> pd.DataFrame or gpd.GeoDataFrame:
    """
    A function to clean a DataFrame from duplicates based on specified columns.

    Args:
        data (pd.DataFrame): The input DataFrame to be cleaned.
        id_column (str): The name of the column to use as the unique identifier.

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: A cleaned DataFrame or GeoDataFrame without duplicates based on the
        specified text column.
    """

    uniq_df = data.drop_duplicates(subset=[id_column], keep="first")
    uniq_df = uniq_df.reset_index(drop=True)

    return uniq_df


@staticmethod
def clean_from_digits(data: pd.DataFrame or gpd.GeoDataFrame, text_column: str) -> pd.DataFrame or gpd.GeoDataFrame:
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
        cleaned_text = "".join([j for j in text if not j.isdigit()])

        data.at[i, text_column] = cleaned_text

    return data


@staticmethod
def clean_from_toponyms(
    data: pd.DataFrame or gpd.GeoDataFrame, text_column: str, name_column: str, toponym_type_column: str
) -> pd.DataFrame or gpd.GeoDataFrame:
    """
    Clean the text in the specified text column by removing any words that match the toponyms in the name
    and toponym columns.

    Args:
        data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text to be cleaned.
        name_column (str): The name of the column containing the toponym name (e.g. Nevski, Moika etc).
        toponym_type_column (str): The name of the column containing the toponym type
        (e.g. street, alley, avenue etc.).

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: The DataFrame or GeoDataFrame with the cleaned text.
    """

    for i in range(len(data)):
        text = str(data[text_column].iloc[i]).lower()
        word_list = text.split()
        toponyms = [str(data[name_column].iloc[i]).lower(), str(data[toponym_type_column].iloc[i]).lower()]

        text = " ".join([j for j in word_list if j not in toponyms])

        data.at[i, text_column] = text

    return data


@staticmethod
def clean_from_links(data: pd.DataFrame or gpd.GeoDataFrame, text_column: str) -> pd.DataFrame or gpd.GeoDataFrame:
    """
    Clean the text in the specified text column by removing links and specific patterns.

    Args:
        data (pd.DataFrame or gpd.GeoDataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text to be cleaned.

    Returns:
        pd.DataFrame or gpd.GeoDataFrame: The DataFrame with the cleaned text.
    """
    for i in range(len(data)):
        text = str(data[text_column].iloc[i])
        if "[id" in text and "]" in text:
            start = text.index("[")
            stop = text.index("]")

            text = text[:start] + text[stop:]

        text = re.sub(r"^https?://.*[\r\n]*", "", text, flags=re.MULTILINE)

        data.at[i, text_column] = text

    return data


@staticmethod
def fill_empty_toponym(data: pd.DataFrame or gpd.GeoDataFrame, toponym_column: str):
    for i in range(len(data)):
        check = data[toponym_column].iloc[i]
        if check == "":
            data.at[i, toponym_column] = None

    return data
