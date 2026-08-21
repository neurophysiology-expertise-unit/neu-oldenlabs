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


def test_check_join_returns_export_only_tags_instead_of_raising(tmp_path):
    # The join is asymmetric (study owner's ruling: the CSV is the sole
    # authority on which animals exist). A tag only in the export -- not
    # listed in the CSV -- is not a real failure: it is returned as an
    # excluded tag for the caller to keep out of analysis and report,
    # never raised on. This replaces the old symmetric rule, under which
    # this same call used to raise "only in export".
    table = load_animals(write_csv(tmp_path))
    assert check_join(table, 58616, {2, 5}) == []
    assert check_join(table, 58616, {2, 5, 9}) == [9]


def test_check_join_still_raises_when_a_csv_listed_animal_is_absent_from_export(tmp_path):
    # The reverse direction is a genuine discrepancy -- the study is missing
    # an animal's entire record -- and remains fatal.
    table = load_animals(write_csv(tmp_path))
    with pytest.raises(ValueError, match="only in animal table"):
        check_join(table, 58616, {2})
