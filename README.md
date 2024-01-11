# SLOYKA

![Your logo](data/photo_2023-12-12_09-56-55.jpg)
[![Documentation Status](https://readthedocs.org/projects/soika/badge/?version=latest)](https://soika.readthedocs.io/en/latest/?badge=latest)
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/scikit-learn/)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Цель проекта

**SLOYKA** - это библиотека, нацеленная на обогащение цифровых моделей городов данными, получаемыми из текстовых данных цифрового следа горожан, а также на моделирование вернакулярной оценки качества городской среды. Она основывается на методах обработки естественных языков (NLP) и методах работы с графами. Основным ее элементом является конструируемый пространственно-семантический граф, пополняемый при помощи машинного распознавания городских сущностей и локаций.

SLOYKA включает в себя две группы методов: методы для генерации пространственно-семантического графа и методы для моделирования социальных процессов с его помощью.

**Модуль генерации пространственного-семантического графа**

Пространственно-семантический граф состоит из двух взаимосвязанных компонентов: 
- Пространственный граф, отображающий географическую близость различных именованных сущностей на территории города (улиц, организаций, парков, точек притяжения и др.)
- Семантический граф, отображающий смысловую и фактологическую близость различных городских сущностей (природных явлений, объектов городской среды, городских пользователей).

Пространственный граф строится по следующему алгоритму:
1. Пользователь подает на вход набор текстов и указание на территорию
2. Из OSM загружается сеть УДС в виде графа и именованные сущности
3. От именованных сущностей строится евклидово расстояние до ближайшего узла графа, оно пишется в новое ребро
4. В текстах выявляются урбанонимы (городские названия). Процесс выполняется при помощи дообученной
[модели](https://huggingface.co/Geor111y/flair-ner-addresses-extractor) RuBert и набора эвристик для геокодирования без использования сторонних сервисов
1. Сначала они геокодируются по адресной системе, потом по названиям объектов. Также учитываются варианты народных написаний
2. Для событий в тексте определяется вероятность локализации. Она может быть увеличена при появлении новых текстов для тех же событий
3. Процесс может выполняться периодично, в таком случае появляется возможность определения активных и латентных территорий

Сравнение результата геокодирования с существующим решением:
![Geocoder comparison](/data/Сравнение_эффективности_извлечения_адреса_3.png)

Выделение комьюнити на основе жалоб горожан по различным тематикам:
![Spatial graph communities](/data/photo_2023-12-25_01-44-15.jpg)

Процентное соотношение жалоб по категориям среди выделенных комьюнити:
![Spatial graph communities](/data/photo_2023-12-25_01-44-14.jpg)


Семантический граф строится по следующему алгоритму:
1. Пользователь подает на вход набор текстов и указание на территорию (Санкт-Петербург)
2. Из текста извлекаются сущности, их контекст, добавляются в узлы
3. Определяется семантическая близость сущностей, наиболее близкие объединяются
4. Дополнительно выделяются опорные узлы-урбанонимы
5. Формируются фактологические связи в виде ребер
6. Сущности не удаляются из графа с течением времени, но имеют период активности. Повторное упоминание обновляет его

![Semantic graph](/data/photo_2024-01-11_20-31-29.jpg)

**Модуль моделирования социальных процессов с помощью пространственного-семантического графа**
- В процессе разработки -

Предполагаемые сценарии использования методов:
- Отслеживание социально значимых ситуаций
- Прогнозирование динамики семантического графа
- Нахождение городских сообществ
- Определение идентичности мест
- Определение вернакулярных районов
- Определение точек притяжения


## Table of Contents

- [SLOYKA](#sloyka)
  - [Цель проекта](#цель-проекта)
  - [Table of Contents](#table-of-contents)
  - [Особенности SLOYKA](#особенности-sloyka)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Examples](#examples)
  - [Documentation](#documentation)
  - [Developing](#developing)
  - [License](#license)
  - [Contacts](#contacts)
  - [Citation](#citation)


## Особенности SLOYKA

- Готовый к использованию инструмент для исследователей и аналитиков, работающих с неструктурированными социальными данными. Наша библиотека поможет извлечь факты из текстов, описывающих городские процессы и явления
- Модульная структура библиотеки позволяет получать и использовать только необходимые части, например, если ваша единственная цель - обогащение модели города пространственными данными о дорожно-транспортных происшествиях
- Эта библиотека может быть использована для моделирования и анализа социальных процессов в городе на основе текстовых данных, выявляя их смысловые и пространственные параметры

## Installation

All details about first steps with SLOYKA might be found in the [install guide](https://soika.readthedocs.io/en/latest/soika/installation.html)
and in the [tutorial for novices](https://soika.readthedocs.io/en/latest/soika/quickstart.html)

## Project Structure

The latest stable release of SOIKA is on the [master branch](https://github.com/GeorgeKontsevik/sloyka/tree/master) 

The repository includes the following directories:

* Package [core](https://github.com/iduprojects/SOIKA/tree/master/factfinder)  contains the main classes and scripts. It is the *core* of SOIKA;
* Package [examples](https://github.com/iduprojects/SOIKA/tree/master/examples) includes several *how-to-use-cases* where you can start to discover how SOIKA works;
* All *unit and integration tests* can be observed in the [test]() directory;
* The sources of the documentation are in the [docs](https://github.com/iduprojects/SOIKA/tree/master/docs) 
                                                        
## Examples
Вы можете использовать свои собственные данные, но они должны соответствовать спецификации. Следующие примеры помогут освоиться с библиотекой:

1. [Geocoder](examples/geocoder_example.ipynb) - как извлечь топонимы и получить их координаты

## Documentation

We have a [documentation](https://soika.readthedocs.io/en/latest/?badge=latest), but our [examples](#examples) will explain the use cases cleaner.

## Developing

To start developing the library, one must perform following actions:

1. Clone repository (`git clone https://github.com/GeorgeKontsevik/sloyka.git`)
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

## Contacts

- [NCCR](https://actcognitive.org/o-tsentre/kontakty) - National Center for Cognitive Research
- [IDU](https://idu.itmo.ru/en/contacts/contacts.htm) - Institute of Design and Urban Studies
- If you have questions or suggestions, you can contact us at the following address: asantonov@itmo.ru (Aleksandr Antonov)

## Citation

---


