"""
This class is aimed to aggregate data by region and provide some information about it users
activity
"""

import pandas as pd
import geopandas as gpd

from sloyka import Geocoder, TextClassifiers, City_services


class RegionalActivity:

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

        self.geocoder = Geocoder()
        self.data = data
        self.processed_geodata = None
        self.osm_id = osm_id
        self.tags = tags
        self.date = date
        self.text = text_column
        self.group_name = group_name_column
        self.text_classifier = TextClassifiers(repository_id=repository_id,
                                               number_of_categories=number_of_categories,
                                               device_type=device)

    def geocode_data(self) -> None:
        """
        This function geocodes the data and returns the comments with the points and addresses
        Returns:
            None
        """

        geocoded_data = self.geocoder.run(df=self.data,
                                          osm_id=self.osm_id,
                                          tags=self.tags,
                                          date=self.date,
                                          text_column=self.text,
                                          group_column=self.group_name)

        self.processed_geodata = geocoded_data

    def service_data(self) -> None:
        """
        This function extracts service from texts
        Returns:
            None
        """

        service_data = City_services().run(df=self.processed_geodata,
                                           text_column=self.text)

        self.processed_geodata = service_data

    def class_text(self) -> None:
        """
        This functions classifies texts according to extracted function
        Returns:
            None
        """

        self.processed_geodata[['classified_text', 'probs']] = (self.processed_geodata[self.text].progress_map(
                                                                lambda x: self.text_classifier.run_text_classifier(x)).
                                                                to_list())


if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\thebe\\OneDrive - ITMO UNIVERSITY\\НИРМА\\Data\\test_data\\regional_activity\\messages.csv")

    ra = RegionalActivity(data=df,
                          osm_id=338635,
                          tags={"admin_level": ["8"]},
                          date="2024-04-22T00:00:00Z",
                          text_column='text')

    ra.geocode_data()
    ra.service_data()
    print(ra.processed_geodata)
