from hypothesis import given, settings, strategies as st

from dask.sizeof import sizeof


@given(st.dictionaries(st.integers(), st.integers(), min_size=11, max_size=30))
@settings(max_examples=200)
def test_sizeof_dict_consistency(d):
    result1 = sizeof(d)
    result2 = sizeof(d)
    assert result1 == result2, f"sizeof(dict) should be deterministic, but got {result1} != {result2}"


if __name__ == "__main__":
    test_sizeof_dict_consistency()