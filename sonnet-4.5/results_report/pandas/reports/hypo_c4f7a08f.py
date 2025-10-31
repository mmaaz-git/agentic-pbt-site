import pandas as pd
from hypothesis import given, strategies as st


@st.composite
def int_arrow_arrays(draw, min_size=0, max_size=30):
    data = draw(st.lists(
        st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
        min_size=min_size,
        max_size=max_size
    ))
    return pd.array(data, dtype='int64[pyarrow]')


@given(arr=int_arrow_arrays(min_size=1, max_size=20))
def test_take_empty_indices(arr):
    """Taking with empty indices should return empty array."""
    result = arr.take([])
    assert len(result) == 0
    assert result.dtype == arr.dtype


if __name__ == "__main__":
    # Run the test
    test_take_empty_indices()