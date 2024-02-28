from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
from natasha.extractors import Extractor
import natasha.obj as obj
import pandas as pd
from natasha.extractors import Match
from .rule_for_natasha import ADDR_PART
import pdb
import requests
import geopandas as gpd
from shapely.geometry import Point
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import math

class AddrNEWExtractor(Extractor):
    def __init__(self, morph):
        Extractor.__init__(self, ADDR_PART, morph)

    def find(self, text):
        matches = self(text)
        if not matches:
            return

        matches = sorted(matches, key=lambda _: _.start)
        if not matches:  
            return
        start = matches[0].start
        stop = matches[-1].stop
        parts = [_.fact for _ in matches]
        return Match(start, stop, obj.Addr(parts))

class NER_parklike:
    @staticmethod
    def extract_parklike(text):
        """
        The function extracts parklike entities in the text, using the Natasha library.
        """
        morph = MorphVocab()
        extractor = AddrNEWExtractor(morph)

        data = {'start': "", 'stop': "", 'toponim': "", 'toponim_type': ""}

        match = extractor.find(text)
        if not match:
            return pd.Series([data['start'], data['stop'], data['toponim'], data['toponim_type']])

        data['start'] = match.start
        data['stop'] = match.stop
        data['toponim'] = match.fact.parts[0].value
        data['toponim_type'] = match.fact.parts[0].type
        return pd.Series([data['start'], data['stop'], data['toponim'], data['toponim_type']])
        

    @staticmethod
    def natasha_normalization(text, start_ind, stop_ind):
        """
        This function extracts entities and normalize them. For entities that match the 
        extraction of "extract_parklike", it returns a normalized form.
        """
        normalized_form = ""
        if start_ind != "":
            return normalized_form
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)
        for span in doc.spans:
            if span.start == start_ind or span.stop == stop_ind:
                span.normalize(morph_vocab)
                normalized_form = span.normal
        return normalized_form
    
    @staticmethod
    def OSM_park_import(city_name):
        query = """
        [out:json];
        area["name"="Санкт-Петербург"]->.a;
        (
            node["leisure"="park"](area.a);
            way["leisure"="park"](area.a);
            relation["leisure"="park"](area.a);
            node["leisure"="garden"](area.a);
            way["leisure"="garden"](area.a);
            relation["leisure"="garden"](area.a);
        );
        out center;
        """

        overpass_url = "http://overpass-api.de/api/interpreter"

        response = requests.get(overpass_url, params={"data": query})
        data = response.json()

        data_list = []
        for element in data["elements"]:
            park_name = element.get("tags", {}).get("name")
            
            if park_name:
                if "center" in element:
                    lat, lon = element["center"]["lat"], element["center"]["lon"]
                elif "lat" in element and "lon" in element:
                    lat, lon = element["lat"], element["lon"]
                else:
                    lat, lon = None, None

                data_list.append({"ParkName": park_name, "Latitude": lat, "Longitude": lon})

        osm_df = pd.DataFrame(data_list)
        geometry = [Point(xy) for xy in zip(osm_df['Longitude'], osm_df['Latitude'])]
        osm_df = gpd.GeoDataFrame(osm_df, geometry=geometry, crs="EPSG:4326")

        osm_df['Geopoint'] = osm_df['geometry'].apply(lambda geom: f'POINT ({geom.x} {geom.y})')

        osm_df = osm_df.drop(['Latitude', 'Longitude', 'geometry'], axis=1)
        return osm_df
    
    @staticmethod
    def split_parkwords(osm_df):
        osm_df['missing_parkword'] = ''
        parkword_list = ['сад', 'сквер', 'аллея', 'парк', 'лесопарк', 'двор']
        
        for index, row in osm_df.iterrows():
            park_name = str(row['ParkName']).lower() 
            
            for parkword in parkword_list:
                if parkword.lower() in park_name:
                    osm_df.at[index, 'ParkName'] = park_name.replace(parkword, '').strip()
                    osm_df.at[index, 'missing_parkword'] = parkword
                    break 
        return osm_df

    @staticmethod
    def cut_park_name(cell_value):
        if cell_value == "":
            return cell_value
        words = cell_value.split()
        cuted_words = [word[:-2] if len(word) > 4 else word for word in words]
        cuted_value = ' '.join(cuted_words)
        return cuted_value
    
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def match_point_name(osm_df, df):
        for index, row in df.iterrows():
            toponim_cut_value = row['Toponim_cut'].lower()
            toponim_type_value = row['Toponim_type'].lower()
            
            for index_osm_df, row_osm_df in osm_df.iterrows():
                processed_park_name_value = row_osm_df['ProcessedParkName'].lower()
                missing_parkword_value = row_osm_df['missing_parkword'].lower()
                
                similarity_toponim_cut = NER_parklike.similar(toponim_cut_value, processed_park_name_value)
                similarity_toponim_type = NER_parklike.similar(toponim_type_value, missing_parkword_value)
                
                if similarity_toponim_cut > 0.8 and similarity_toponim_type > 0.2:
                    df.at[index, 'ParkNameFull'] = row_osm_df['ParkNameFull']
                    df.at[index, 'Geopoint'] = row_osm_df['Geopoint']
                    break
        return df



    def run(self, df, text_column, city_name):
        df[["Start", "Stop", "Toponim", "Toponim_type"]] = df[text_column].apply(lambda text: NER_parklike.extract_parklike(text) if NER_parklike.extract_parklike(text) is not None else None)
        osm_df = NER_parklike.OSM_park_import(city_name)
        osm_df['ParkNameFull']= osm_df['ParkName']
        osm_df = NER_parklike.split_parkwords(osm_df)
        osm_df['ProcessedParkName'] = osm_df['ParkName'].apply(NER_parklike.cut_park_name)
        df['Toponim_cut'] = df['Toponim'].apply(NER_parklike.cut_park_name)
        df = NER_parklike.match_point_name(osm_df, df)
        # df["Normal"] = df.apply(lambda row: NER_parklike.natasha_normalization(row[text_column], row["Start"], row["Stop"]),axis=1)
        return df
        

        
                     
