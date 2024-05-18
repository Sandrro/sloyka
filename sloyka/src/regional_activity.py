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
                 text_column: str = 'text',
                 group_name_column: str = 'group_name',
                 group_region_column: str = 'territory',
                 admin_level_column: str = 'admin_level'
                 ) -> None:
        """
        This function initializes the RegionalActivity class by
        Args:
            group_region_column: The name of the region to be analyzed
        """
        self.data = data
        self.geodata = None
        self.text = text_column
        self.group_name = group_name_column
        self.group_region = group_region_column
        self.admin_level = admin_level_column

    def geocode_data(self) -> gpd.GeoDataFrame:
        """
        This function geocodes the data and returns the comments with the points and addresses
        Returns:
            gpd.GeoDataFrame
        """
        geocoded_data = gpd.GeoDataFrame()
        groups = self.data[self.group_name].unique()

        for i in groups:
            data_info = self.data.loc[self.data[self.group_name] == i]

            city = data_info[self.group_region].iloc[0]
            level = int(data_info[self.admin_level].iloc[0])

            sloyka_geocoder = Geocoder(osm_city_name=city, osm_city_level=level)

            tmp_df = self.data.loc[self.data[self.group_region] == i]
            tmp_gdf = sloyka_geocoder.run(tmp_df, text_column=self.text)

            geocoded_data = gpd.GeoDataFrame(pd.concat([geocoded_data, tmp_gdf]), index=geocoded_data.index)

            self.geodata = geocoded_data

        return geocoded_data


if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\thebe\\OneDrive - ITMO UNIVERSITY\\НИРМА\\Data\\test_data\\regional_activity\\messages.csv")

    ra = RegionalActivity(data=df)
    print(ra.data)
    print(ra.geocode_data())
