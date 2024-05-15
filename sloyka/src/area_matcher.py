import re
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.stem.snowball import SnowballStemmer
from sloyka.src.data_getter import HistGeoDataGetter
stemmer = SnowballStemmer("russian")

class AreaMatcher:
    def __init__(self):
        self.area_cache = {}

    def get_df_areas(self, osm_id):
        if osm_id not in self.area_cache:
            tags = {"admin_level": ["5","6","8"]}
            date = "2024-04-22T00:00:00Z"
            geo_data_getter = HistGeoDataGetter()
            df_areas = geo_data_getter.get_features_from_id(osm_id=osm_id, tags=tags, date=date)
            df_areas = df_areas[df_areas['element_type'] != 'way']
            self.area_cache[osm_id] = df_areas
        return self.area_cache[osm_id]

    def match_area(self, group_name, osm_id):
        df_areas = self.get_df_areas(osm_id)

        group_name = group_name.lower()
        group_name = re.sub(r'[\"!?\u2665\u2022()|,.-:]', '', group_name)
        words_to_remove = ["сельское поселение", "городское поселение", "район", "округ", "город"]
        for word in words_to_remove:
            df_areas['area_name'] = df_areas['name'].str.replace(word, '', regex=True)

        words_to_remove_f_groups = [
            "сельское поселение", "городское поселение",
            "район", "округ", "город", "муниципальное образование",
            " МО ", "МО ", "Муниципальный"]
        for word in words_to_remove_f_groups:
            group_name = re.sub(word, '', group_name, flags=re.IGNORECASE)

        group_name_stems = [stemmer.stem(word) for word in group_name.split()]
        max_partial_ratio = 20
        max_token_sort_ratio = 20
        best_match = None
        
        has_digits = any(char.isdigit() for char in group_name)
        
        if has_digits:
            for _, row in df_areas.iterrows():
                if row['area_name'].isdigit():
                    if row['area_name'] in group_name:
                        best_match = row['area_name']
                        break
        
        if best_match:
            return best_match
        
        for _, row in df_areas.iterrows():
            area = row['area_name'].lower()
            area = re.sub(r'[\"!?\u2665\u2022()|,.-:]', '', area)
            area_stems = [stemmer.stem(word) for word in area.split()]
            
            partial_ratio = fuzz.partial_ratio(group_name, area)
            token_sort_ratio = fuzz.token_sort_ratio(group_name_stems, area_stems)
            
            if partial_ratio > max_partial_ratio and token_sort_ratio > max_token_sort_ratio:
                max_partial_ratio = partial_ratio
                max_token_sort_ratio = token_sort_ratio
                best_match = row['area_name']
        
        return best_match