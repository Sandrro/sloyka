import pandas as pd
from transformers import pipeline


class TextClassifierTopics:
    """
    This class is aimed to classify input texts into themes, or structured types of events. It uses a Huggingface transformer model trained on rubert-tiny.
    In many cases count of messages per theme was too low to efficiently train, so we used synthetic themes based on the categories as upper level (for example, 'unknown_ЖКХ')
    """

    def __init__(
        self,
        repository_id="Sandrro/text_to_subfunction_v10",
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
        This method takes a text as input and returns the predicted themes and probabilities.
        :param t: text to classify
        :return: list of predicted themes and probabilities
        """
        preds = pd.DataFrame(self.classifier(t, top_k=self.CATS_NUM))
        self.classifier.call_count = 0
        if self.CATS_NUM > 1:
            cats = "; ".join(preds["label"].tolist())
            probs = "; ".join(preds["score"].round(3).astype(str).tolist())
        else:
            cats = preds["label"][0]
            probs = preds["score"].round(3).astype(str)[0]
        return [cats, probs]
