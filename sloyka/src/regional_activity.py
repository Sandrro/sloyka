"""
This class is aimed to aggregate data by region and provide some information about it users
activity
"""
from typing import Union, Optional
import pandas as pd
import geopandas as gpd

from sloyka.src.geocoder.geocoder import Geocoder
from sloyka.src.text_classifiers import TextClassifiers
from sloyka.src.utils.data_processing.city_services_extract import City_services
from sloyka.src.risks.emotion_classifier import EmotionRecognizer


class RegionalActivity:
    """This class is aimed to produce a geodataframe with the main information about users activity.
    It uses other sloyka modules such as Geocoder, TextClassifiers, City_services and EmotionRecognizer to process data.
    Processed data is saved in class attribute 'processed_geodata' and
    after class initialization can be called with RegionalActivity.processed_geodata.
    

    Args:
        data (pd.DataFrame): DataFrame with posts, comments and replies in text format
        with additional information such as 
        date, group_name, text_type, parents_id and so on.
        Expected to be formed from sloyka.VKParser.run class function output.
        osm_id (int): OSM ID of the place from which geograhic data should be retrieved.
        tags (dict): toponyms_dict with tags to be used in the Geocoder
        (e.g. {'admin_level': [5, 6]}).
        date (str): Date from wheach data from OSM should be retrieved.
        path_to_save (str, optional): Path to save processed geodata. Defaults to None.
        text_column (str, optional): Name of the column with text in the data. Defaults to 'text'.
        group_name_column (str, optional): Name of the column with group name in the data.
        Defaults to 'group_name'.
        repository_id (str, optional): ID of the repository to be used in TextClassifiers.
        Defaults to 'Sandrro/text_to_subfunction_v10'.
        number_of_categories (int, optional): Number of categories to be used in TextClassifiers.
        Defaults to 1.
        device (str, optional): Device type to be used in models. Defaults to 'cpu'.
        use_geocoded_data (bool, optional): Whether the input data is geocoded or not. If True skips geocoding fase. Defaults to False.
    """

    def __init__(self,
                 data: Union[pd.DataFrame, gpd.GeoDataFrame],
                 osm_id: int,
                 tags: dict[str, list[int]],
                 date: str,
                 path_to_save: Optional[str] = None,
                 text_column: str = 'text',
                 group_name_column: str = 'group_name',
                 repository_id: str = 'Sandrro/text_to_subfunction_v10',
                 number_of_categories: int = 1,
                 device: str='cpu',
                 use_geocoded_data: bool = False
                 ) -> None:

        self.data: pd.DataFrame | gpd.GeoDataFrame = data
        self.osm_id: int = osm_id
        self.tags: dict[str, list[int]] = tags
        self.date: str = date
        self.text: str = text_column
        self.group_name: str = group_name_column
        self.device: str = device
        self.path_to_save: str | None = path_to_save
        self.use_geocoded_data: bool = use_geocoded_data
        self.text_classifier = TextClassifiers(repository_id=repository_id,
                                               number_of_categories=number_of_categories,
                                               device_type=device)
        self.processed_geodata: gpd.GeoDataFrame = self.run_sloyka_modules()
        self.top_topics = self.processed_geodata.copy()['cats'].value_counts(normalize=True)[:5]*100
        
    def run_sloyka_modules(self) -> gpd.GeoDataFrame:
        """This function runs data with the main functions of the Geocoder, TextClassifiers,City_services and
        EmotionRecognizer classes. If path_to_save was provided it also saves data in the path.

        Returns:
            None: Data is saved in RegionalActivity.processed_geodata and written to the path if path_to_save was provided
        """
        
        if self.use_geocoded_data:
            processed_geodata: gpd.GeoDataFrame = self.data.copy() # type: ignore
        else:
            processed_geodata: gpd.GeoDataFrame = Geocoder(device=self.device,
                                                           osm_id=self.osm_id,
                                                           city_tags=self.tags).run(
                df=self.data,
                text_column=self.text,
                group_column=self.group_name,
                search_for_objects=True
            ) # type: ignore
        
        processed_geodata[['cats',
                           'probs']] = processed_geodata[self.text].progress_map(
                               lambda x: self.text_classifier.run_text_classifier(x)).to_list() # type: ignore
        processed_geodata = City_services().run(df=processed_geodata,
                                                     text_column=self.text)
        processed_geodata = EmotionRecognizer().add_emotion_column(df=processed_geodata,
                                                                   text=self.text)

        if self.path_to_save:
            processed_geodata.to_file(self.path_to_save)

        return processed_geodata

    def update_geodata(self, data:Union[pd.DataFrame, gpd.GeoDataFrame]) -> None:
        """
        Update the geodata with the given data. Will ran all stages for the new data.
        If path_to_save was provided it also saves data in the path.
        If use_geocoded_data was set to True, it will skip the geocoding stage.

        Args:
            data: The data to update the geodata with.

        Returns:
            None
        """

        self.data = data
        self.processed_geodata = self.run_sloyka_modules()

    @staticmethod
    def get_chain_ids(name: str,
                      data: Union[pd.DataFrame, gpd.GeoDataFrame],
                      id_column: str,
                      name_column: str) -> list:
        """This function creates a tuple of unique identifiers of chains of posts, comments and
        replies around specified value in column.

        Args:
            name (str): value in column to select a post-comment-reply chains
            data (Union[pd.DataFrame, gpd.GeoDataFrame]): data with posts, comments and replies
            id_column (str): column with unique identifiers
            name_column (str): column to base a selection

        Returns:
            tuple: tuple of unique identifiers of chains
        """
        
        posts_ids = data[id_column].loc[data[name_column] == name].to_list()
        comments_ids = data[id_column].loc[data['post_id'].isin(posts_ids) & data[name_column].isin([name, None])].to_list()
        replies_ids = data[id_column].loc[data['parents_stack'].isin(comments_ids) & data[name_column].isin([name, None])].to_list()

        return tuple(sorted(list(set(posts_ids + comments_ids + replies_ids))))
    
    @staticmethod
    def dict_to_df(toponyms_dict: dict) -> pd.DataFrame:
        """This function converts dictionary created by RegionalActivity.get_risks() function to a DataFrame object.

        Args:
            toponyms_dict (dict): dictionary with info created in RegionalActivity.get_risks().

        Returns:
            pd.DataFrame: Table with info about top n most mentioned toponyms.
        """
        
        df_list = []
        
        for toponym in toponyms_dict:
            for service in toponyms_dict[toponym]['services']:
                for emotion in toponyms_dict[toponym]['services'][service]['emotions']:
                    df_list.append([toponym,
                                    toponyms_dict[toponym]['part_users'],
                                    toponyms_dict[toponym]['part_messages'],
                                    service,
                                    toponyms_dict[toponym]['services'][service]['counts'],
                                    emotion,
                                    toponyms_dict[toponym]['services'][service]['emotions'][emotion]])
                    
        dataframe = pd.DataFrame(df_list, columns=['Toponym', 'Part_users', 'Part_messages', 'Service', 'Counts', 'Emotion', 'Emotion_count'])
        
        return dataframe

    def get_risks(self,
                  processed_data: Optional[gpd.GeoDataFrame]=None,
                  top_n: int=5,
                  to_df: bool=False) -> dict:
        """This function returns a toponyms_dict with info about top n most mentioned toponyms.
        toponyms_dict have the following format:
        {'Toponym': {'part_users' : 0.0,
                     'part_messages' : 0.0,
                     'services' : {'service_1' : {counts : 0,
                                                  emotions: {'emotion' : 0.0}
                                                  },
                                   }
                     }
         }

        Args:
            top_n (int, optional): The number of most mentioned toponyms to be calculated. Defaults to 5.
            to_df (bool, optional): Whether to return a DataFrame or a dictionary. Defaults to False.

        Returns:
            dict: toponyms_dict with info about top n most mentioned toponyms with the following format.
            
            {'Toponym': {'part_users' : 0.0,
                        'part_messages' : 0.0,
                        'services' : {'service_1' : {counts : 0,
                                                    emotions: {'emotion' : 0.0}
                                                    },
                                    }
                        }
            } 
        """

        if not processed_data:
            gdf_final = self.processed_geodata.copy()
        else:
            gdf_final = processed_data
        top_n_toponyms = self.processed_geodata['only_full_street_name'].value_counts(normalize=True).index[:top_n]
        
        result = {}

        for i in top_n_toponyms:
            
            
            all_ids = self.get_chain_ids(name=i,
                                         data=gdf_final,
                                         id_column='id',
                                         name_column='only_full_street_name')
            
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
                
            if to_df:
                result = self.dict_to_df(result)
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
