from hypothesis import given, strategies as st, settings, assume
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@given(
    st.integers(min_value=2, max_value=20),
    st.sampled_from(['D', '2D', '3D', 'H', '2H', 'W', 'M']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
)
@settings(max_examples=100)  # Reduced for faster testing
def test_no_duplicate_divisions(n_divisions, rule, closed, label):
    start = pd.Timestamp('2020-01-01')
    divisions = pd.date_range(start, periods=n_divisions, freq='D')

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(
            divisions, rule, closed=closed, label=label
        )
    except Exception as e:
        assume(False)

    assert len(outdivs) == len(set(outdivs)), \
        f"outdivs contains duplicates: {outdivs} for n_divisions={n_divisions}, rule={rule}, closed={closed}, label={label}"

if __name__ == "__main__":
    # Run the test
    print("Running property-based test...")
    try:
        test_no_duplicate_divisions()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Also test the specific failing case mentioned
    print("\nTesting specific failing case: n_divisions=2, rule='2D', closed='left', label='left'")
    start = pd.Timestamp('2020-01-01')
    divisions = pd.date_range(start, periods=2, freq='D')
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, '2D', closed='left', label='left')
    print(f"outdivs: {outdivs}")
    print(f"Has duplicates: {len(outdivs) != len(set(outdivs))}")