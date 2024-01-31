import pytest
from factfinder.src.data_getter import VkCommentsParser


@pytest.mark.parametrize(
        "post_id, owner_id, token, result_len",
        [
            ("10796", -51988192, '96cbbc1496cbbc1496cbbc14b795dfa8b8996cb96cbbc14f2a8294fa4c4c6fc7e753a93', 2)
        ],
)

def test_get_comments(post_id, owner_id, token, result_len):
    test_df = VkCommentsParser.get_Comments(post_id, owner_id, token)
    assert len(test_df) == result_len