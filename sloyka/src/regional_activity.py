"""
This class is aimed to aggregate data by region and provide some information about it users
activity
"""

import pandas as pd
import geopandas as gpd

from sloyka.src.geocoder import Geocoder
from sloyka.src.text_classifiers import TextClassifiers 
from sloyka.src.city_services_extract import City_services 
from sloyka.src.emotionclass import EmotionRecognizer


class RegionalActivity:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        osm_id (int): _description_
        tags (dict): _description_
        date (str): _description_
        text_column (str, optional): _description_. Defaults to 'text'.
        group_name_column (str, optional): _description_. Defaults to 'group_name'.
        repository_id (str, optional): _description_. Defaults to 'Sandrro/text_to_subfunction_v10'.
        number_of_categories (int, optional): _description_. Defaults to 1.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 osm_id: int,
                 tags: dict,
                 date: str,
                 path_to_save: str = None,
                 text_column: str = 'text',
                 group_name_column: str = 'group_name',
                 repository_id: str = 'Sandrro/text_to_subfunction_v10',
                 number_of_categories: int = 1,
                 device: str='cpu'
                 ) -> None:

        self.data = data
        self.osm_id = osm_id
        self.tags = tags
        self.date = date
        self.text = text_column
        self.group_name = group_name_column
        self.device = device
        self.text_classifier = TextClassifiers(repository_id=repository_id,
                                               number_of_categories=number_of_categories,
                                               device_type=device)
        self.processed_geodata = self.run_sloyka_modules()
        self.top_topics = self.processed_geodata.copy()['cats'].value_counts(normalize=True)[:5] * 100
        
        if path_to_save:
            self.path_to_save = path_to_save
        
    def run_sloyka_modules(self) -> gpd.GeoDataFrame:
        """This function 

        Returns:
            None: No data provided
        """
        
        processed_geodata = Geocoder(device=self.device).run(df=self.data,
                                                             osm_id=self.osm_id,
                                                             tags=self.tags,
                                                             date=self.date,
                                                             text_column=self.text,
                                                             group_column=self.group_name)
        
        processed_geodata[['cats', 'probs']] = (processed_geodata[self.text].progress_map(
                                                                lambda x: self.text_classifier.run_text_classifier(x)).
                                                                to_list())
        
        processed_geodata = City_services().run(df=processed_geodata,
                                                     text_column=self.text)
        
        processed_geodata = EmotionRecognizer().add_emotion_column(df=processed_geodata,
                                                                   text=self.text)
        
        if self.path_to_save:
            processed_geodata.to_file(self.path_to_save)
        
        return processed_geodata

    def get_risks(self,
                  top_n: int=5) -> dict:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """

        gdf_final = self.processed_geodata.copy()
        
        top_n_toponyms = self.processed_geodata['only_full_street_name'].value_counts(normalize=True).index[:top_n]
        
        result = {}

        for i in top_n_toponyms[:5]:
            
            posts_ids = gdf_final['id'].loc[gdf_final['only_full_street_name'] == i].to_list()
            comments_ids = gdf_final['id'].loc[gdf_final['post_id'].isin(posts_ids) & gdf_final['only_full_street_name'].isin([i, None])].to_list()
            replies_ids = gdf_final['id'].loc[gdf_final['parents_stack'].isin(comments_ids) & gdf_final['only_full_street_name'].isin([i, None])].to_list()
            
            all_ids = tuple(sorted(list(set(posts_ids + comments_ids + replies_ids))))
            
            toponym_gdf_final = gdf_final.loc[gdf_final['id'].isin(all_ids)]
            part_users = len(toponym_gdf_final['from_id'].unique())/len(gdf_final['from_id'].unique())
            part_messages = len(toponym_gdf_final['id'].unique())/len(gdf_final['id'].unique())
            
            info = {'part_users': part_users,
                    'part_messages': part_messages}
            
            services = [obj for inner_list in toponym_gdf_final['City_services'].to_list() for obj in inner_list]
            
            if services:
                unique_services = set(services)
                if unique_services:
                    values = [0 for j in unique_services]
                    
                    services_dict = dict.fromkeys(unique_services, 0)
                    
                    service_info = {}
                    for j in services:
                        if j in services_dict:
                            services_dict[j] = services_dict[j] + 1
                
                service_gdf = toponym_gdf_final.dropna(subset='City_services')
                
            else:
                continue
                
            for j in unique_services:
                
                service_info = {'counts': 0}
                
                if j in services_dict:
                        service_info['counts'] = service_info['counts'] + 1
                
                emotions = service_gdf['emotion'].to_list()
                
                if emotions:
                    unique_emotions = set(emotions)
                    values = [0 for k in unique_emotions]

                    emotions_dict = {key: value for key, value in zip(unique_emotions, values)}
                    
                    for k in emotions:
                        if k in emotions_dict:
                            emotions_dict[k] = emotions_dict[k] + 1
                    
                service_info['emotions'] = emotions_dict
                services_dict[j] = service_info
                info['services'] = services_dict
                result[i] = info
                
        return result
        

if __name__ == "__main__":
    
    df = pd.read_csv('',
                     sep=';',
                     index_col=0)

    ra = RegionalActivity(data=df,
                          osm_id=338635,
                          tags={"admin_level": ["8"]},
                          date="2024-04-22T00:00:00Z",
                          text_column='text')

    print(ra.processed_geodata['Location'], ra.processed_geodata['City_services'], ra.processed_geodata['classified_text'])
    print(ra.get_risks())
