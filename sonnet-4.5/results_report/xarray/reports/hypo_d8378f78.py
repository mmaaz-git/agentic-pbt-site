from hypothesis import given, strategies as st, assume, settings
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    num=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=200)
def test_linspace_with_endpoint(start, stop, num):
    assume(start != stop)

    index = RangeIndex.linspace(
        start=start,
        stop=stop,
        num=num,
        endpoint=True,
        dim="x"
    )
    assert index.size == num

# Run the test
test_linspace_with_endpoint()