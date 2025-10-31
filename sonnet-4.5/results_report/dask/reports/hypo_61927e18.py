from hypothesis import given, strategies as st
from dask.dataframe.utils import valid_divisions

@given(st.lists(st.integers(), max_size=1))
def test_valid_divisions_small_lists(divisions):
    try:
        result = valid_divisions(divisions)
        assert isinstance(result, bool), f"Should return bool, got {type(result)}"
    except IndexError:
        assert False, f"Should not crash on {divisions}"

if __name__ == "__main__":
    test_valid_divisions_small_lists()