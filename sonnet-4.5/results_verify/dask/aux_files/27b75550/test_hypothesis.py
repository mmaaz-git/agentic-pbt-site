#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings, example
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=2, max_value=20),
    st.sampled_from(['h', 'D', '2h', '3D', '12h', 'W']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
)
@example(7, 'W', 'left', 'left')  # The specific failing case
@settings(max_examples=100)  # Reduced for quick testing
def test_resample_divisions_monotonic(n_divs, freq, closed, label):
    start = pd.Timestamp('2000-01-01')
    end = start + pd.Timedelta(days=30)
    divisions = pd.date_range(start, end, periods=n_divs)

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, freq, closed=closed, label=label)

    # Check strictly monotonic (no duplicates allowed)
    for i in range(len(outdivs) - 1):
        assert outdivs[i] < outdivs[i+1], f"outdivs not monotonic at {i}: {outdivs[i]} >= {outdivs[i+1]}, params: n_divs={n_divs}, freq={freq}, closed={closed}, label={label}"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_resample_divisions_monotonic()
        print("✓ All tests passed!")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")