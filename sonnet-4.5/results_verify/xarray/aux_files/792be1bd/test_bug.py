from hypothesis import given, strategies as st, assume, settings
from xarray.indexes.range_index import RangeIndex

@given(
    start=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    stop=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    step=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
)
@settings(max_examples=1000)
def test_arange_size_nonnegative(start, stop, step):
    assume(step != 0)
    assume(abs(step) > 1e-10)

    index = RangeIndex.arange(start, stop, step, dim="x")
    assert index.size >= 0, f"Size must be non-negative, got {index.size}"

if __name__ == "__main__":
    test_arange_size_nonnegative()