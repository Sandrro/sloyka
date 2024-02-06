import pandas as pd
from transformers import pipeline


class TextClassifier:
    """
    This class is aimed to classify input texts into categories, or city functions. It uses a Huggingface transformer model trained on rubert-tiny
    """

    def __init__(
        self,
        repository_id="Sandrro/text_to_function_v2",
        number_of_categories=1,
        device_type=None,
    ):
        self.REP_ID = repository_id
        self.CATS_NUM = number_of_categories
        self.classifier = pipeline(
            "text-classification",
            model=self.REP_ID,
            tokenizer="cointegrated/rubert-tiny2",
            max_length=2048,
            truncation=True,
            device=device_type,
        )

    def run(self, t):
        """
        This method takes a text as input and returns the predicted categories and probabilities.
        :param t: text to classify
        :return: list of predicted categories and probabilities
        """
        if isinstance(t, str):
            preds = pd.DataFrame(self.classifier(t, top_k=self.CATS_NUM))
            self.classifier.call_count = 0
            if self.CATS_NUM > 1:
                cats = "; ".join(preds["label"].tolist())
                probs = "; ".join(preds["score"].round(3).astype(str).tolist())
            else:
                cats = preds["label"][0]
                probs = preds["score"].round(3).astype(str)[0]
        else:
            print("text is not string")
            cats = None
            probs = None
        return [cats, probs]
