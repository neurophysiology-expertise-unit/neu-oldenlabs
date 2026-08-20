from __future__ import annotations

import pytest

from neu_oldenlabs.animals import check_join, load_animals

CSV = """cage #,animal #,,genotype,tag #,Tag date,Monitoring start
58616,659,L,W/w,2,260706,
,660,R,W/w,5,,
58623,668,L,W/w,3,,
,669,R,Mut/W,1,,
,672,RR,W/W,4,,
"""


def write_csv(tmp_path, text=CSV):
    path = tmp_path / "animals.csv"
    path.write_text(text)
    return path


def test_cage_is_forward_filled_and_types_are_integers(tmp_path):
    table = load_animals(write_csv(tmp_path))
    assert list(table["cage"]) == [58616, 58616, 58623, 58623, 58623]
    assert table["tag"].tolist() == [2, 5, 3, 1, 4]
    assert table["animal_number"].dtype.kind == "i"


def test_upper_and_lower_case_wildtype_both_normalise_to_wt(tmp_path):
    table = load_animals(write_csv(tmp_path))
    by_number = table.set_index("animal_number")
    assert by_number.loc[672, "genotype"] == "wt"
    assert by_number.loc[672, "genotype_raw"] == "W/W"
    assert by_number.loc[659, "genotype"] == "wt"
    assert by_number.loc[669, "genotype"] == "mut"


def test_unknown_genotype_string_raises(tmp_path):
    bad = CSV.replace("Mut/W", "mut/w ?")
    with pytest.raises(ValueError, match="unknown genotype"):
        load_animals(write_csv(tmp_path, bad))


def test_duplicate_tag_within_a_cage_raises(tmp_path):
    bad = CSV.replace(",660,R,W/w,5,,", ",660,R,W/w,2,,")
    with pytest.raises(ValueError, match="duplicate tag"):
        load_animals(write_csv(tmp_path, bad))


def test_check_join_requires_identical_tag_sets(tmp_path):
    table = load_animals(write_csv(tmp_path))
    check_join(table, 58616, {2, 5})
    with pytest.raises(ValueError, match="only in export"):
        check_join(table, 58616, {2, 5, 9})
    with pytest.raises(ValueError, match="only in animal table"):
        check_join(table, 58616, {2})
