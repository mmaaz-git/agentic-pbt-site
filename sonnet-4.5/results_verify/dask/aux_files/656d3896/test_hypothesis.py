from hypothesis import given, strategies as st, assume, settings
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def date_range_divisions(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    num_periods = draw(st.integers(min_value=2, max_value=100))
    freq = draw(st.sampled_from(['h', 'D', '2h', '30min', 'W']))
    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    divisions = pd.date_range(start=start, periods=num_periods, freq=freq)
    return tuple(divisions)


@given(
    divisions=date_range_divisions(),
    rule=st.sampled_from(['h', 'D', '2D', '30min', 'W', '2W']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=100)  # Reduced for quick test
def test_resample_bin_and_out_divs_monotonic_increasing(divisions, rule, closed, label):
    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    except Exception as e:
        assume(False)  # Skip if function raises an exception

    # Check if outdivs is monotonically increasing
    for i in range(len(outdivs) - 1):
        assert outdivs[i] < outdivs[i+1], \
            f"outdivs should be monotonically increasing at index {i}: {outdivs[i]} >= {outdivs[i+1]}"

    # Also check newdivs
    for i in range(len(newdivs) - 1):
        assert newdivs[i] < newdivs[i+1], \
            f"newdivs should be monotonically increasing at index {i}: {newdivs[i]} >= {newdivs[i+1]}"

# Run the test
print("Running hypothesis test...")
try:
    test_resample_bin_and_out_divs_monotonic_increasing()
    print("✓ All tests passed!")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")