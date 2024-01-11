import os
import pandas as pd
import gensim
from nltk.tokenize import sent_tokenize, regexp_tokenize
import re
from gensim.models import word2vec
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec
from bs4 import BeautifulSoup
import csv
import networkx as nx
from string import ascii_letters as al

nltk.download("punkt")
nltk.download("stopwords")


class SemanticValue:
    """
    This class is created for making a Word2Vec semantic model in order to evaluate semantic closeness from nx graph.
    """

    def __init__(self):
        """Initialization function with main data paths and tokinizer."""
        self.tokenizer = nltk.data.load("tokenizers/punkt/russian.pickle")
        self.current_dir = os.path.dirname(
            os.path.abspath(os.path.join(__file__, ".."))
        )
        self.models_path = os.path.join(self.current_dir, "models")
        self.data_path = os.path.join(self.current_dir, "training_data")
        self.file_path = os.path.join(self.data_path, "comments_vk.txt")

    def change_data_path(self, name: str) -> None:
        """Change file path."""
        self.file_path = name

    def review(self, review: str) -> str:
        """Function clean string from links."""
        return re.sub(
            r"http[s]?://(?:[f-za-z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fa-f][0-9a-fa-f]))+",
            " ",
            review,
        )

    def review_to_wordlist(self, review: str, remove_stopwords=False) -> str:
        """Function makes a list of words from string sentense."""
        review = self.review(review)
        review_text = BeautifulSoup(review, "lxml").get_text()
        review_text = re.sub("[^а-яа-я]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = stopwords.words("russian")
            words = [w for w in words if not w in stops]
        return words

    def review_to_sentences(self, review: str, remove_stopwords=False) -> list:
        """Function makes a list of sentences from list of words."""
        raw_sentences = self.tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(
                    self.review_to_wordlist(raw_sentence, remove_stopwords)
                )
        return sentences

    def clean_and_convert_csv(
        self, data_name: str = "posts_spb_today.csv"
    ) -> list:
        """Functions cleans csv and converts it to training set."""
        with open(self.file_path, "a", encoding="utf-8") as file:
            with open(
                os.path.join(self.data_path, data_name), "r", encoding="utf-8"
            ) as data:
                for row in data:
                    file.write(row)

        with open(self.file_path, encoding="utf-8") as f:
            data = "".join(i for i in f.read() if not i.isdigit())

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(data)

        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        text = re.sub("\n", " ", text)
        sents = sent_tokenize(text)

        punct = '!"#$%&()*+,-./:;<=>?@[\\]^_{|}~„“«»†*—/\\-‘’'
        clean_sents = []

        for sent in sents:
            s = [w.lower().strip(punct) for w in sent.split()]
            clean_sents.append(s)

        return clean_sents

    def train_model(
        self,
        custom_data: str = "posts_spb_today.csv",
        column_name: str = "text",
        model_name: str = "default_trained.model",
    ) -> None:
        """Function creates and trains a model from custom data."""
        clean_sents = self.clean_and_convert_csv(data_name=custom_data)

        model = word2vec.Word2Vec(
            clean_sents, vector_size=10, window=5, min_count=1
        )

        data = pd.read_csv(
            os.path.join(self.data_path, custom_data), delimiter=";"
        )
        sentences = []

        print("Parsing sentences from training set...")
        for review in data[column_name]:
            sentences += self.review_to_sentences(review, self.tokenizer)

        custom_model = word2vec.Word2Vec(
            sentences,
            workers=-1,
            vector_size=300,
            min_count=10,
            window=10,
            sample=1e-3,
        )

        custom_model_path = os.path.join(self.models_path, model_name)

        print("Saving model...")
        custom_model.save(custom_model_path)

        model_trained = gensim.models.Word2Vec.load(custom_model_path)
        model_trained.build_vocab(clean_sents, update=True)
        model_trained.train(
            sentences, total_examples=model_trained.corpus_count, epochs=5
        )

    def expectlatin(self, mystring: str) -> str:
        """Function checks for latin words. Temporary sollution."""
        result = ""
        for i in mystring:
            if i not in al:
                result += i
        if result != "":
            return result
        else:
            return "абв"

    def evaluate_graph(self, graph_file: str = "kg.graphml") -> None:
        """Function evaluates words from nx graph and makes a csv file with evaluation results."""

        custom_model_path = os.path.join(
            self.models_path, "default_trained.model"
        )

        # open model
        model_trained = gensim.models.Word2Vec.load(custom_model_path)

        graph_path = os.path.join(self.data_path, graph_file)

        G = nx.read_graphml(graph_path)
        df = nx.to_pandas_edgelist(G)
        # create and write csv
        output_file_path = os.path.join(
            os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))),
            "output_data",
        )
        csv_path = os.path.join(output_file_path, "semantic_value.csv")

        with open(csv_path, mode="w", encoding="utf-8") as file:
            # intiate counters
            sucess = 0
            fail = 0

            # establish columns names
            names = ["source", "target", "semantic closeness"]

            # create a writer
            file_writer = csv.DictWriter(
                file, delimiter=";", lineterminator="\r", fieldnames=names
            )

            for i in range(len(df)):
                # make a list of pair words
                words = [df["source"][i].lower(), df["target"][i].lower()]
                # cleaning words in created list
                words = [
                    self.expectlatin("".join(c for c in word if c.isalpha()))
                    for word in words
                ]

                for word in words:
                    # checking for word in model
                    if word in model_trained.wv.key_to_index:
                        check_status = True
                    else:
                        fail += 1
                        check_status = False
                        break

                # checking results and writing it in csv
                if check_status == True:
                    sucess += 1
                    custom_result = model_trained.wv.similarity(
                        words[0], words[1]
                    )
                    file_writer.writerow(
                        {
                            "source": words[0],
                            "target": words[1],
                            "semantic closeness": custom_result,
                        }
                    )

            print(f"Succesfully evaluated: {sucess}\nFailed: {fail}")


if __name__ == "__main__":
    processor = SemanticValue()
    processor.train_model()
    processor.evaluate_graph("kg.graphml")
