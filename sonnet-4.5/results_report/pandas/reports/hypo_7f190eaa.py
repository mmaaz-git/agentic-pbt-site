from pandas.util.version import Infinity, NegativeInfinity
from hypothesis import given, strategies as st


@given(st.just(Infinity))
def test_infinity_reflexive_comparisons(inf):
    assert inf == inf
    assert inf <= inf
    assert inf >= inf
    assert not (inf < inf)
    assert not (inf > inf)


@given(st.just(NegativeInfinity))
def test_negative_infinity_reflexive_comparisons(ninf):
    assert ninf == ninf
    assert ninf <= ninf
    assert ninf >= ninf
    assert not (ninf < ninf)
    assert not (ninf > ninf)


if __name__ == "__main__":
    test_infinity_reflexive_comparisons()
    test_negative_infinity_reflexive_comparisons()