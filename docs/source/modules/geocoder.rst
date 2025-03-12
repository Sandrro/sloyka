.. _geocoder:

Geocoding
==================
To perform classification with generalized linear models, see :ref:`Geocoder_special`.

Geocoder
-------------------
.. autoclass:: soika.src.geocoder.geocoder.Geocoder
   :members: run

OtherGeoObjects
---------------------
.. autoclass:: soika.src.geocoder.city_objects_extractor.OtherGeoObjects
   :members: run

StreetExtractor
---------------------
.. autoclass:: soika.src.geocoder.street_extractor.StreetExtractor
   :members: process_pipeline

more:
-------------------------------------
.. toctree::
   :maxdepth: 1
   :caption: Advanced geocoding

   Geocoder_special
   OtherGeoObjects
   StreetExtractor