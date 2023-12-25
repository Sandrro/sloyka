"""
These are constants that are used in many factfinder files.
"""

global_crs: int = 4326
device: str = "cpu"
osm_city_level: int = 5
osm_city_name: str = "Санкт-Петербург"
model_path: str = "Geor111y/flair-ner-addresses-extractor"
target_score: int = 0.7
end_index_position: int = 4
start_index_position: int = 3
target_toponyms = ["пр", "проспект", "проспекте", "ул", "улица", "улице", "площадь", "площади", "пер", "переулок", "проезд", "проезде", "дорога", "дороге"]
replacement_dict = {
            "пр": "проспект",
            "ул": "улица",
            "пер": "переулок",
            "улице": "улица",
            "проспекте": "проспект",
            "площади": "площадь",
            "проезде": "проезд",
            "дороге": "дорога"
}
