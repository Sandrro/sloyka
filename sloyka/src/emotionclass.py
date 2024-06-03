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

    def recognize_emotion(self, text):
        """Return the emotion for a given text."""
        emotion = self.recognizer.recognize(text, return_single_label=True)
        return emotion

    def recognize_average_emotion_from_multiple_models(self, text, models=None):
        """Calculate the prevailing emotion using multiple models."""
        default_models = [
            HuggingFaceModel.Text.Bert_Tiny,
            HuggingFaceModel.Text.Bert_Base,
            HuggingFaceModel.Text.Bert_Large,
            HuggingFaceModel.Text.Bert_Tiny2
        ]

        if models is None:
            models = default_models
        else:
            # Validate that the provided models are in the default models list
            for model in models:
                if model not in default_models:
                    raise ValueError(f"Model {model} is not a valid model. Valid models are: {default_models}")

        scores = {emotion: 0 for emotion in ["happiness", "sadness", "anger", "fear", "disgust", "enthusiasm", "neutral"]}

        recognizers = [TextRecognizer(model=model, device=self.device) for model in models]

        for recognizer in recognizers:
            results = recognizer.recognize(text, return_single_label=False)
            for emotion, score in results.items():
                scores[emotion] += score
        
        # Average the scores by the number of models
        for emotion in scores:
            scores[emotion] /= len(recognizers)

        # Determine the prevailing emotion with the highest average score
        prevailing_emotion = max(scores, key=scores.get)
        return prevailing_emotion