Intro to SOIKA
===============


**SOIKA** is a library designed to achieve the following objectives:

- Identification of points of public activity and related objects within the urban environment.
- Forecasting social risks associated with objects of public activity.
- Utilizing natural language analysis (NLP) methods to analyze text messages from citizens on open communication platforms.

The library comprises several modules, including classification, geolocation, and risk detection.

The Structure of the Modeling Pipeline
----------------------------------------

.. image:: /docs/img/pipeline_en.png
   :alt: Pypline

In the modeling pipeline, SOIKA is organized as follows:

Classification Module
~~~~~~~~~~~~~~~~~~~~~

The classification module employs a cascading approach with two classification methods:

1. The first method relies on a pre-trained spaCy model (which will be replaced with BERT in the future) to determine the primary city function affected by the problem described in a complaint. Examples of city functions include communal services and environmental protection.

2. The second method assigns complaints to predefined topic clusters. These clusters are generated using a topic modeling algorithm for each city function and are evaluated by experts in these respective functions.

Geolocation Module
~~~~~~~~~~~~~~~~~~

The geolocation module utilizes a ruBERT-based method for message geolocation. It combines a pre-trained Named Entity Recognition (NER) model to extract location details (district, street, and house number) from text with approximate string matching to assign coordinates from the Open Data portal OpenStreetMaps to the extracted location information.

Risk Detection Module
~~~~~~~~~~~~~~~~~~~~

The risk detection module processes data obtained from the previous modules. Texts with timestamps, spatial characteristics, and functional attributes are input into a multi-level algorithm based on BERTopic. This process results in a collection of interrelated situations. The potential risk associated with these situations can be assessed using additional data.
