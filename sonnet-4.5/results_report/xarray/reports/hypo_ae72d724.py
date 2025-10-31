from hypothesis import given, strategies as st, assume, settings
from xarray.indexes import RangeIndex


@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_arange_step_nonzero(start, stop, step):
    assume(step != 0)
    assume(abs(step) > 1e-10)
    assume((stop - start) / step < 1e6)
    assume((stop - start) / step > -1e6)

    index = RangeIndex.arange(
        start=start,
        stop=stop,
        step=step,
        dim="x"
    )
    assert index.size >= 0

# Run the test
test_arange_step_nonzero()