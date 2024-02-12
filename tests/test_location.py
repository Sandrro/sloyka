
from sloyka.src.geocoder import Geocoder

# def test_geocode_with_retry(input_address, geocode_result):
#     result = Location().geocode_with_retry(input_address)
#     assert result.address == geocode_result


# def test_geocode_with_retry_empty_address():
#     result = Location().geocode_with_retry("")
#     assert result is None


# def test_query(input_address, geocode_result):
#     result = Location().query(input_address)
#     assert result.address == geocode_result


@pytest.mark.parametrize(
    "input_address,geocode_result",
    [
        ("возле дома на Итальянской 17 постоянно мусорят!!!", "Итальянской 17"),
    ],
)
def test_geolocator(input_address, geocode_result):
    result = Geocoder().extract_ner_street(input_address)
    assert result.loc[0] == geocode_result
