from hypothesis import given, strategies as st, settings
from pandas.util.version import InfinityType, NegativeInfinityType


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=200)
def test_infinity_comparison_consistency(n):
    inf1 = InfinityType()
    inf2 = InfinityType()

    if inf1 == inf2:
        assert not (inf1 > inf2)
        assert not (inf1 < inf2)
        assert inf1 <= inf2
        assert inf1 >= inf2


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=200)
def test_negative_infinity_comparison_consistency(n):
    ninf1 = NegativeInfinityType()
    ninf2 = NegativeInfinityType()

    if ninf1 == ninf2:
        assert not (ninf1 > ninf2)
        assert not (ninf1 < ninf2)
        assert ninf1 <= ninf2
        assert ninf1 >= ninf2


if __name__ == "__main__":
    test_infinity_comparison_consistency()
    test_negative_infinity_comparison_consistency()