
from sloyka import Geocoder
import pytest

@pytest.mark.parametrize(
    "input_address,geocode_result",
    [
        ("возле дома на Итальянской 17 постоянно мусорят!!!", "Итальянской 17"),
    ],
)
def test_geolocator(input_address, geocode_result):
    result = Geocoder().extract_ner_street(input_address)
    assert result.loc[0] == geocode_result
