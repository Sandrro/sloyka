import re
import osmnx as ox
import geopandas as gpd
import pandas as pd
import time
from osmnx._errors import InsufficientResponseError as NoResponse

tags = {
    "subway": ["yes"],
    "amenity": ["university", "school"],
    "landuse": [ "brownfield", "cemetery", "commercial", "construction", "flowerbed",
    "grass", "industrial", "meadow", "military", "plant_nursery",
    "recreation_ground", "religious", "residential", "retail"],
    "natural": ["water", "beach"],
    "leisure": ["garden","marina","nature_reserve","park","pitch","sports_centre"],
    "highway": ["construction","footway","motorway","pedestrian","primary","primary_link","residential","secondary","service","steps","tertiary","tertiary_link","unclassified"],
    "railway": ["rail", "subway"],
    "amenity": ["arts_centre","atm","bank","bar","boat_rental","bus_station","bicycle_rental","biergarten","cafe","car_wash","childcare","cinema","clinic","clinic;doctors;audiologist","college","community_centre","courthouse","coworking_space","dancing_school","dentist","doctors","driving_school","events_venue","fast_food","fire_station","food_court","fountain","fuel","hookah_lounge","hospital","internet_cafe","kindergarten","language_school","library","music_school","music_venue","nightclub","offices","parcel_locker","parking","payment_centre","pharmacy","place_of_worship","police","post_office","pub","recycling","rescue_station","restaurant","school","social_centre","social_facility","studio","theatre","training","university","vending_machine","veterinary","townhall","shelter","marketplace","monastery","planetarium","research_institute"],
    "building": ["apartments","boat","bunker","castle","cathedral","chapel","church","civic","college","commercial","detached","dormitory","garages","government","greenhouse","hospital","hotel","house","industrial","kindergarten","kiosk","mosque","office","pavilion","policlinic","public","residential","retail","roof","ruins","school","service","ship","sport_centre","sports_hall","theatre","university"],
    "man_made": ["bridge","courtyard","lighthouse","mineshaft","pier","satellite","tower","works"],
    "leisure": ["amusement_arcade","fitness_centre","playground","sauna","stadium","track"],
    "office": ["company","diplomatic","energy_supplier","government","research","telecommunication"],
    "shop": ["alcohol","antiques","appliance","art","baby_goods","bag","bakery","bathroom_furnishing","beauty","beauty;hairdresser;massage;cosmetics;perfumery","bed","beverages","bicycle","binding","bookmaker","books","boutique","butcher","car","car_parts","car_repair","carpet","cheese","chemist","clothes","coffee","computer","confectionery","convenience","copyshop","cosmetics","cosmetics;chemist","craft","craft;paint","curtain","dairy","deli","doityourself","doors","dry_cleaning","e-cigarette","electrical","electronics","electronics;fishing","erotic","estate_agent","fabric","farm","fireplace","fishing","flooring","florist","frame","frozen_food","furniture","games","garden_centre","gas","general","gift","glaziery","gold_buyer","greengrocer","hairdresser","hairdresser_supply","hardware","health_food","hearing_aids","herbalist","honey","houseware","interior_decoration","jeweller_tools","jewelry","kiosk","kitchen","laundry","leather","lighting","lottery","massage","medical_supply","mobile_phone","money_lender","motorcycle","music","newsagent","nuts","optician","orthopaedic","orthopaedics","outdoor","outpost","paint","pastry","pawnbroker","perfumery","pet","pet_grooming","photo","pottery","printer_ink","printing","radiotechnics","repair","seafood","second_hand","security","sewing","shoes","sports","stationery","stationery;copyshop","storage_rental","supermarket","tableware","tailor","tattoo","tea","ticket","tobacco","toys","travel_agency","variety_store","watches","water_filter","weapons","wine"],
    "bus": ["yes"],
    "public_transport": ["platform","station","stop_position"],
    "railway": ["tram_stop","station"]
}

gdf_list = []
metric_crs = 3857

if __name__ == "__main__":
        
    place_name = input("Input the place name (Default value: 'Saint-Petersburg, Russia'):")

    if place_name.strip() == "":
        place_name = "Saint-Petersburg, Russia"
        
    epsg = input("Input the Coordinate Reference System (Default value: 'EPSG:4326'):")

    if epsg.strip() == "":
        epsg = 'EPSG:4326'
        
    crs = int(re.sub("[^0-9]", "", epsg))

    csv = input("Write results to CSV files? (y/n):")
    print_csv = True if (csv.strip().lower() == 'y' or epsg.strip().lower() == "yes") else False
    if print_csv == True:
        import os
        os.chdir(os.path.dirname(__file__))


    tot_cat = len(tags.keys())
    tot_tag = sum([len(values) for values in tags.values()])
    cnt_cat, cnt_tag = 0, 0

    for category, category_tags in tags.items():
        
        cnt_cat += 1
        
        for tag in category_tags:
            cnt_tag += 1
            try:
                tmpgdf = ox.features_from_place(place_name, tags={category: tag})
                if len(tmpgdf) < 10:
                    tmpgdf = None
                    raise NoResponse
                tmpgdf.crs = epsg
                tmpgdf = tmpgdf.to_crs(crs=metric_crs)
                tmpgdf['tag'] = category
                tmpgdf['centroid'] = tmpgdf["geometry"].centroid
                gdf = tmpgdf
                gdf.geometry = tmpgdf.buffer(0.001)
                gdf = gdf.to_crs(crs=crs)
                tmpgdf = None
                
                print(f'{time.strftime("[%Y-%m-%d %H-%M-%S]")} Completed Tag "{category}" #{cnt_cat} out of {tot_cat}; Key "{tag}" #{cnt_tag} out of {tot_tag}.')
                
                if print_csv == True:
                    gdf.to_csv(f'{category}-{tag}_{time.strftime("D%Y-%m-%dT%H-%M-%S")}.csv')
                    
                gdf_list.append(gdf)
                
            except (NoResponse, AttributeError):
                print(f'{time.strftime("[%Y-%m-%d %H-%M-%S]")} No Data for Tag "{category}" #{cnt_cat} out of {tot_cat}; Key "{tag}" #{cnt_tag} out of {tot_tag}.')
            

            

    print(f'{time.strftime("[%Y-%m-%d %H-%M-%S]")} Done processing. Merging results...') 
    merged_gdf = pd.concat(gdf_list)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    selected_columns = ['tag','element_type', 'osmid', 'name', 'geometry', 'centroid']
    limited_gdf = merged_gdf.reset_index()[selected_columns]
    
    limited_gdf
    input(f'{time.strftime("[%Y-%m-%d %H-%M-%S]")} Done. Press ENTER to quit...') 
