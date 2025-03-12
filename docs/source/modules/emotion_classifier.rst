.. _emotion_classifier:

Emotion classifier
==================

.. automodule:: soika.src.risks.emotion_classifier
   :members:
   :undoc-members:

Example
-------
.. code-block:: bash

   df = pd.read_csv('data.csv')
   recognizer = EmotionRecognizer()
   df['emotion'] = df['text'].apply(recognizer.recognize_emotion)
