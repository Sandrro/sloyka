"""
This module contains the EmotionClassifiers class, which is designed to categorise input texts into emotion categories.
It uses a Huggingface transformer model trained on Bert_Large by default.
The EmotionClassifiers class has the following method:

@method:recognize_emotion: Adding an emotion category.

@method:recognize_average_emotion_from_multiple_models: Adding an average emotion category using multiple models.
"""

from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel
import torch
import pandas as pd
from tqdm import tqdm

class EmotionRecognizer:
    """
    This class is designed to categorise input texts into emotion categories.

        Attributes:

    - model: This attribute holds the model used for emotion recognition. It defaults to HuggingFaceModel.Text.Bert_Large, 
    but can be set to any other compatible model during the instantiation of the class.

    - device: the device to use for inference. It automatically selects 'cuda' (GPU) if a compatible GPU 
    is available and CUDA is enabled, otherwise, it falls back to 'cpu'.

    - text: The text to be analyzed.
    """

    def __init__(self, model=HuggingFaceModel.Text.Bert_Large, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.recognizer = TextRecognizer(model=self.model, device=self.device)

        # Preload all models to avoid repeated initialization
        self.default_models = [
            HuggingFaceModel.Text.Bert_Tiny,
            HuggingFaceModel.Text.Bert_Base,
            HuggingFaceModel.Text.Bert_Large,
            HuggingFaceModel.Text.Bert_Tiny2
        ]

    def recognize_emotion(self, text):
        """Return the emotion for a given text."""
        emotion = self.recognizer.recognize(text, return_single_label=True)
        return emotion

    def _recognize_with_model(self, model, texts):
        """Helper function to recognize emotion with a given model for a list of texts."""
        recognizer = TextRecognizer(model=model, device=self.device)
        results = [recognizer.recognize(text, return_single_label=False) for text in texts]
        return results
    
    def recognize_average_emotion_from_multiple_models(self, df, text_column, models=None, average=True):
        """Calculate the prevailing emotion using multiple models for a DataFrame column."""
        if models is None:
            models = self.default_models
        else:
            for model in models:
                if model not in self.default_models:
                    raise ValueError(f"Model {model} is not a valid model. Valid models are: {self.default_models}")

        scores = pd.DataFrame(0, index=df.index, columns=["happiness", "sadness", "anger", "fear", "disgust", "enthusiasm", "neutral"])
        counts = pd.DataFrame(0, index=df.index, columns=["happiness", "sadness", "anger", "fear", "disgust", "enthusiasm", "neutral"])

        for model in tqdm(models, desc="Processing models"):
            model_results = self._recognize_with_model(model, df[text_column])

            for idx, result in enumerate(model_results):
                for emotion, score in result.items():
                    scores.at[df.index[idx], emotion] += score
                    counts.at[df.index[idx], emotion] += 1

        if average:
            scores = scores.div(len(models))
            prevailing_emotions = scores.idxmax(axis=1)
        else:
            prevailing_emotions = counts.idxmax(axis=1)

        return prevailing_emotions