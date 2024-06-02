"""
This class is aimed to aggregate data by region and provide some information about it users
activity
"""

import pandas as pd
import geopandas as gpd

from sloyka import Geocoder, TextClassifiers, City_services


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
        self.text_classifier = TextClassifiers(repository_id=repository_id,
                                               number_of_categories=number_of_categories,
                                               device_type=device)
        self.processed_geodata = Geocoder().run(df=self.data,
                                                osm_id=self.osm_id,
                                                tags=self.tags,
                                                date=self.date,
                                                text_column=self.text,
                                                group_column=self.group_name)
        self.processed_geodata = City_services().run(df=self.processed_geodata,
                                                     text_column=self.text)
        self.processed_geodata[['classified_text', 'probs']] = (self.processed_geodata[self.text].progress_map(
                                                                lambda x: self.text_classifier.run_text_classifier(x)).
                                                                to_list())
        
        self.top_topics = self.processed_geodata.copy()['cats'].value_counts(normalize=True)[:5] * 100

    def get_risks(self,
                  top_n: int=5) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """

        top_n_toponyms = self.processed_geodata['only_full_street_name'].value_counts(normalize=True).index[:top_n]
        
        res_dict = {}
        
        for i in top_n_toponyms:
            
            posts_ids = df['id'].loc[df['toponym'] == i].to_list()
            comments_ids = df['id'].loc[df['post_id'].isin(posts_ids) & df['toponym'].isin([i, None])].to_list()
            replies_ids = df['id'].loc[df['parents_stack'].isin(comments_ids) & df['toponym'].isin([i, None])].to_list()
            
            all_ids = tuple(sorted(list(set(posts_ids + comments_ids + replies_ids))))
            
            toponym_df = df.loc[df['id'].isin(all_ids)]
            part_users = len(toponym_df['from_id'].unique())/len(df['from_id'].unique())
            part_messages = len(toponym_df['id'].unique())/len(df['id'].unique())
            
            services = df['City_services'].loc[df['toponym'] == i]
            
            res_dict[i] = {'part_users': part_users,
                           'part_messages': part_messages,
                           'services': services}
        

if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\thebe\\OneDrive - ITMO UNIVERSITY\\НИРМА\\Data\\test_data\\regional_activity\\comments.csv")

    ra = RegionalActivity(data=df,
                          osm_id=338635,
                          tags={"admin_level": ["8"]},
                          date="2024-04-22T00:00:00Z",
                          text_column='text')

    print(ra.processed_geodata['Location'], ra.processed_geodata['City_services'], ra.processed_geodata['classified_text'])
