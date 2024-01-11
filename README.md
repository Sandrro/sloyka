# SOIKA

![Your logo](https://i.ibb.co/qBjVx8N/soika.jpg)


[![Documentation Status](https://readthedocs.org/projects/soika/badge/?version=latest)](https://soika.readthedocs.io/en/latest/?badge=latest)
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/scikit-learn/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## The purpose of the project

**SOIKA** is a library aimed at identifying points of public activity and related objects of the urban environment, as well as forecasting social risks associated with objects of public activity, based on the application of natural language analysis (NLP) methods to text messages of citizens in open communication platforms. 

SOIKA consists of several modules: classification, geolocation and risk detection.

**Classification module**

For the module, a pre-designated set of citizens’ appeals on urban issues was used. Based on the set, the [BERT model](https://huggingface.co/cointegrated/rubert-tiny) was additionally trained for classification. The final f-score is 0.79. The marked classes include such city functions and industries as housing and communal services, landscaping, safety, transport, roads, healthcare, education, social protection, construction, waste management, ecology, and energy.

[Link to the model](https://huggingface.co/Sandrro/text_to_function_v2)

**Geolocation module**

Named-entity recognition (NER) is used to extract addresses from messages. The [BERT (rubert) model](https://huggingface.co/cointegrated/rubert-tiny) was chosen for training, which allowed taking into account the context of the sentence before and after the search word. The flair framework is used to define named entities. The final f-score of the model is 0.81. The trained model allows obtaining the street and the house number from unstructured text (if specified). Next, toponyms are brought into their initial form and geocoded using Nominatim.

[Link to the model](https://huggingface.co/Geor111y/flair-ner-addresses-extractor)

**Event modeling and risk detection module**

This module is used to generate events. An event is an object that has temporal, spatial, and functional (semantic) characteristics and influences the citizens’ lives. For this reason, to model it, you either need to use the above modules for a set of text data with timestamps, or prepare similar spatial data yourself. To simulate an event, you either need to apply the above-described modules to a set of text data with timestamps, or prepare similar spatial data yourself.

![mat](/docs/img/mathematics.png)

Within the module, an algorithm based on BERTopic is applied to texts linked to a spatial city model at 4 levels: global, street, street section, and building. This makes it possible to track connections between particular cases and the general issue, as well as various events among themselves. Connections are formed on the basis of common texts in different spatial-semantic clusters. At the last stage, the risk of events is calculated as the degree of its impact on the city population.

Risk is calculated using the following formula:  
R = idw, where  
i – intensity (number of distinct text per event)  
d – duration  (number of days when event was active)  
w – significance (mean value of negativity in texts by city functions, calculated by methods of sentiment analysis)


## Table of Contents

- [Core features](#soika-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Documentation](#documentation)
- [Developing](#developing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contacts](#contacts)
- [Citation](#citation)


## SOIKA Features

- Ready-for-use tool for researchers or analytics who work with unstructured social data. Our library can assist with extracting facts from texts from social media without any additional software
- Module structure of the library allows to get and use only neccessary parts, for example, if your only goal is to produce geolocated messages about road accidents
- This library can be used for risk assesment and modelling based on data with functional, spatial and temporal dimensions

## Installation

All details about first steps with SOIKA might be found in the [install guide](https://soika.readthedocs.io/en/latest/soika/installation.html)
and in the [tutorial for novices](https://soika.readthedocs.io/en/latest/soika/quickstart.html)

## Project Structure

The latest stable release of SOIKA is on the [master branch](https://github.com/iduprojects/SOIKA/tree/master) 

The repository includes the following directories:

* Package [core](https://github.com/iduprojects/SOIKA/tree/master/factfinder)  contains the main classes and scripts. It is the *core* of SOIKA;
* Package [examples](https://github.com/iduprojects/SOIKA/tree/master/examples) includes several *how-to-use-cases* where you can start to discover how SOIKA works;
* All *unit and integration tests* can be observed in the [test]() directory;
* The sources of the documentation are in the [docs](https://github.com/iduprojects/SOIKA/tree/master/docs) 
                                                        
## Examples
You are free to use your own data, but it should match specification classes. Next examples will help to get used to the library:

1. [Classifier](examples/classifier_example.ipynb) - how to use city functions classifier
2. [Event detection](examples/event_detection_example.ipynb) - how to generate events from prepared data
3. [Geocoder](examples/geocoder_example.ipynb) - how to extract toponymes and get their coordinates
4. [Topic classifier](examples/topic_classifier_example.ipynb) - how to use classifier for various themes of events
5. [Pipeline example](examples/pipeline_example.ipynb) - how to generate events from texts with use of all of the modules descripted above



## Documentation

We have a [documentation](https://soika.readthedocs.io/en/latest/?badge=latest), but our [examples](#examples) will explain the use cases cleaner.

## Developing

To start developing the library, one must perform following actions:

1. Clone repository (`git clone https://github.com/iduprojects/SOIKA`)
2. (optionally) create a virtual environment as the library demands exact packages versions: `python -m venv venv` and activate it.
3. Install the library in editable mode: `python -m pip install -e '.[dev]' --config-settings editable_mode=strict`
4. Install pre-commit hooks: `pre-commit install`
5. Create a new branch based on **develop**: `git checkout -b develop <new_branch_name>`
6. Add changes to the code
7. Make a commit, push the new branch and create a pull-request into **develop**

Editable installation allows to keep the number of re-installs to the minimum. A developer will need to repeat step 3 in case of adding new files to the library.

A more detailed guide to contributing is available in the [documentation](docs/source/contribution.rst).

## License

The project has [MIT License](./LICENSE)

## Acknowledgments

The library was developed as the main part of the ITMO University project #622264 **"Development of a service for identifying objects of the urban environment of public activity and high-risk situations on the basis of text messages of citizens"**


## Contacts

- [NCCR](https://actcognitive.org/o-tsentre/kontakty) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/en/contacts/contacts.htm) - Institute of Design and Urban Studies
- If you have questions or suggestions, you can contact us at the following address: mvin@itmo.ru (Maxim Natykin)

## Citation

1. B. Nizomutdinov. The study of the possibilities of Telegram bots as a communication channel between authorities and citizens // Proceedings of the 2023 Communication Strategies in Digital Society Seminar (2023 ComSDS) ISBN 979-8-3503-2003-9/23/$31.00 ©2023 IEEE

2. A. Antonov, L. Vidiasova, A. Chugunov. Detecting public spaces and risk situations in them via social media data // Lecture Notes in Computer Science (LNCS), 2023, LNCS 14025, pp. 3–13, 2023. https://doi.org/10.1007/978-3-031-35915-6_1

3. Низомутдинов, Б. А. Тестирование методов обработки комментариев из Telegram-каналов и пабликов ВКонтакте для анализа социальных медиа / Б. А. Низомутдинов, О. Г. Филатова // International Journal of Open Information Technologies. – 2023. – Т. 11, № 5. – С. 137-145. – EDN RWNAOP.


