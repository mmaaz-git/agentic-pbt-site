from hypothesis import given, strategies as st, settings
from xarray.indexes.range_index import RangeIndex

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    num=st.integers(min_value=1, max_value=100),
    endpoint=st.booleans()
)
@settings(max_examples=1000)
def test_linspace_no_crash(start, stop, num, endpoint):
    index = RangeIndex.linspace(start, stop, num=num, endpoint=endpoint, dim="x")
    assert index.size == num

if __name__ == "__main__":
    test_linspace_no_crash()