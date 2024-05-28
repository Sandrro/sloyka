"""
This module contains the EmotionClassifiers class, which is designed to categorise input texts into emotion categories.
It uses a Huggingface transformer model trained on Bert_Large by default.

Attributes:
- model: This attribute holds the model used for emotion recognition. It defaults to HuggingFaceModel.Text.Bert_Tiny2, 
but can be set to any other compatible model during the instantiation of the class.

- device: Determines the computing device for model operations. It automatically selects 'cuda' (GPU) if a compatible GPU 
is available and CUDA is enabled, otherwise, it falls back to 'cpu'. This ensures optimal performance where possible.

- df: The DataFrame (or GeoDataFrame) containing the text data.

- text: The name of the column in df that contains the text to be analyzed. 

The EmotionClassifiers class has the following method:

@method:add_emotion_column: Adding a column with the prevailing emotion category.
"""

import pandas as pd
from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel
import torch

class EmotionRecognizer:
    def __init__(self, model=HuggingFaceModel.Text.Bert_Large):
        # Инициализация модели Bert_Tiny2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.recognizer = TextRecognizer(model=self.model, device=self.device)

    def add_emotion_column(self, df, text):
        """Enhance the data with an emotion column based on text data."""
        # Check if the text column exists
        if text not in df.columns:
            raise ValueError(f"Column '{text}' does not exist in the data.")

        # Apply the recognizer to each row in the text column
        df['emotion'] = df[text].apply(lambda x: self.recognizer.recognize(x, return_single_label=True))
        return df
