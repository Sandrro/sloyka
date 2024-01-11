TARGET_SCORE: int = 0.7
END_INDEX_POSITION: int = 4
START_INDEX_POSITION: int = 3
TARGET_TOPONYMS = ["пр", "проспект", "проспекте", "ул", "улица", "улице", "площадь", "площади", "пер", "переулок", "проезд", "проезде", "дорога", "дороге"]
REPLACEMENT_DICT = {
            "пр": "проспект",
            "ул": "улица",
            "пер": "переулок",
            "улице": "улица",
            "проспекте": "проспект",
            "площади": "площадь",
            "проезде": "проезд",
            "дороге": "дорога"
}
CRS_GLOBAL: int = 4326
GLOBAL_CRS = 4326
GLOBAL_METRIC_CRS = 3857
GLOBAL_EPSG = 'EPSG:4326'

OSM_TAGS = {
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
