from hypothesis import given, strategies as st
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

@given(st.just(None))  # We don't need any parameters, just run once
def test_always_greater_than_total_ordering(dummy):
    """Test that AlwaysGreaterThan satisfies total ordering properties."""
    inf1 = AlwaysGreaterThan()
    inf2 = AlwaysGreaterThan()

    # Test reflexivity: a == a
    assert inf1 == inf1

    # Test symmetry: if a == b then b == a
    assert inf1 == inf2
    assert inf2 == inf1

    # Test antisymmetry: if a == b, then not (a > b) and not (a < b)
    assert inf1 == inf2
    assert not (inf1 != inf2)
    assert not (inf1 < inf2), "AlwaysGreaterThan instances should not be less than each other"
    assert not (inf1 > inf2), f"AlwaysGreaterThan instances should not be greater than each other when equal. Got inf1 > inf2 = {inf1 > inf2}"

    # Test that <= and >= work correctly for equal values
    assert inf1 <= inf2, "inf1 <= inf2 should be True when they are equal"
    assert inf1 >= inf2, "inf1 >= inf2 should be True when they are equal"

@given(st.just(None))  # We don't need any parameters, just run once
def test_always_less_than_total_ordering(dummy):
    """Test that AlwaysLessThan satisfies total ordering properties."""
    ninf1 = AlwaysLessThan()
    ninf2 = AlwaysLessThan()

    # Test reflexivity: a == a
    assert ninf1 == ninf1

    # Test symmetry: if a == b then b == a
    assert ninf1 == ninf2
    assert ninf2 == ninf1

    # Test antisymmetry: if a == b, then not (a > b) and not (a < b)
    assert ninf1 == ninf2
    assert not (ninf1 != ninf2)
    assert not (ninf1 < ninf2), f"AlwaysLessThan instances should not be less than each other when equal. Got ninf1 < ninf2 = {ninf1 < ninf2}"
    assert not (ninf1 > ninf2), "AlwaysLessThan instances should not be greater than each other"

    # Test that <= and >= work correctly for equal values
    assert ninf1 <= ninf2, "ninf1 <= ninf2 should be True when they are equal"
    assert ninf1 >= ninf2, "ninf1 >= ninf2 should be True when they are equal"

if __name__ == "__main__":
    # Run the tests
    test_always_greater_than_total_ordering()
    test_always_less_than_total_ordering()