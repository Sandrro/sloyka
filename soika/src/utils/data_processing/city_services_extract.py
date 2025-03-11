import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
from rapidfuzz import fuzz
from soika.src.utils.constants import CITY_SERVICES_NAMES
from soika.src.utils.data_preprocessing.preprocessor import PreprocessorInput

class City_services:

    def __init__(self, model_name:str="Glebosol/city_services"):
        self.tagger = SequenceTagger.load(model_name)

    def extraction_services(self, text):
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        entities = sentence.get_spans("ner")
        entity_names = [entity.text for entity in entities]
        return entity_names

    def remove_last_letter(words):
        reduced_words = [word[:-1] for word in words]
        return reduced_words

    def replace_with_most_similar(list_of_entities):
        similarity_matrix = np.zeros((len(list_of_entities), len(CITY_SERVICES_NAMES)))
        for i, word1 in enumerate(list_of_entities):
            for j, word2 in enumerate(CITY_SERVICES_NAMES):
                similarity = fuzz.ratio(word1, word2) / 100.0
                similarity_matrix[i, j] = similarity
        new_list_of_entities = list_of_entities.copy()
        for i in range(len(list_of_entities)):
            max_index = np.argmax(similarity_matrix[i])
            new_list_of_entities[i] = CITY_SERVICES_NAMES[max_index]
        return new_list_of_entities

    def run(self, df, text_column):
        df = PreprocessorInput.run(df, text_column)
        city_services = City_services()
        df["City_services_extraced"] = df[text_column].apply(city_services.extraction_services)
        df["City_services_cuted"] = df["City_services_extraced"].apply(City_services.remove_last_letter)
        df["City_services"] = df["City_services_cuted"].apply(City_services.replace_with_most_similar)
        df.drop("City_services_cuted", axis=1, inplace=True)
        return df
