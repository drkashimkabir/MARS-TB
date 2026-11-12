import pytest

from mars_tb.data.label_mapping import category_from_filename, encode_aux, to_binary


@pytest.mark.parametrize(
    "cat,expected",
    [
        ("healthy", 0),
        ("sick_non_tb", 0),
        ("active_tb", 1),
        ("latent_tb", 1),
        ("active_and_latent_tb", 1),
        ("uncertain_tb", None),
    ],
)
def test_binary_mapping_matches_the_manuscript_endpoint(cat, expected):
    assert to_binary(cat) == expected


def test_unknown_category_is_an_error():
    with pytest.raises(ValueError):
        to_binary("pneumonia")


def test_filename_prefix_gives_only_the_coarse_class():
    assert category_from_filename("h0004.png") == "healthy"
    assert category_from_filename("imgs/s0016.png") == "sick_non_tb"
    assert category_from_filename("tb0009.png") == "tb_unspecified"  # active/latent NOT guessed
    assert to_binary("tb_unspecified") == 1


def test_bad_filename_rejected():
    with pytest.raises(ValueError):
        category_from_filename("IMG_0001.png")


def test_aux_labels_missing_become_ignore_index_not_a_class():
    assert encode_aux([None, "", float("nan"), "miliary"], ("cavitary", "infiltrative", "miliary", "pleural")) == [
        -1,
        -1,
        -1,
        2,
    ]
    with pytest.raises(ValueError):
        encode_aux(["bogus"], ("a", "b"))
