from hypothesis import given, strategies as st, settings, example
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@given(
    st.integers(min_value=2, max_value=20),
    st.sampled_from(['h', 'D', '2h', '3D', '12h', 'W']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
)
@example(2, 'h', 'right', 'left')  # The specific failing case mentioned
@settings(max_examples=500)
def test_resample_divisions_contain_original_boundaries(n_divs, freq, closed, label):
    start = pd.Timestamp('2000-01-01')
    end = start + pd.Timedelta(days=30)
    divisions = pd.date_range(start, end, periods=n_divs)

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, freq, closed=closed, label=label)

    assert outdivs[0] >= divisions[0], f"First outdiv {outdivs[0]} before first division {divisions[0]}"


if __name__ == "__main__":
    # Run the test
    test_resample_divisions_contain_original_boundaries()
    print("Test completed successfully - no failures found")