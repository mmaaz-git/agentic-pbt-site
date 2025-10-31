import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, assume
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def date_range_divisions(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))

    periods = draw(st.integers(min_value=2, max_value=100))
    freq_choice = draw(st.sampled_from(['h', 'D', '2h', '6h', '12h', '2D', '3D']))

    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    divisions = pd.date_range(start=start, periods=periods, freq=freq_choice)

    return list(divisions)


@st.composite
def resample_params(draw):
    divisions = draw(date_range_divisions())

    rule = draw(st.sampled_from(['h', '2h', '6h', '12h', 'D', '2D', '3D', 'W', 'ME']))
    closed = draw(st.sampled_from(['left', 'right']))
    label = draw(st.sampled_from(['left', 'right']))

    return divisions, rule, closed, label


@settings(max_examples=200)
@given(resample_params())
def test_resample_divisions_same_length(params):
    divisions, rule, closed, label = params

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    except Exception:
        assume(False)

    assert len(newdivs) == len(outdivs), f"newdivs and outdivs have different lengths: {len(newdivs)} vs {len(outdivs)}"

# Also test the specific failing input mentioned
def test_specific_failing_case():
    from pandas import Timestamp
    divisions = [Timestamp('2000-01-01 00:00:00'), Timestamp('2000-01-01 06:00:00'), Timestamp('2000-01-01 12:00:00')]
    rule = '12h'
    closed = 'right'
    label = 'right'

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)
    print(f"newdivs length: {len(newdivs)}, outdivs length: {len(outdivs)}")
    print(f"newdivs: {newdivs}")
    print(f"outdivs: {outdivs}")

    assert len(newdivs) == len(outdivs), f"newdivs and outdivs have different lengths: {len(newdivs)} vs {len(outdivs)}"

if __name__ == "__main__":
    print("Testing specific failing case...")
    try:
        test_specific_failing_case()
        print("Specific test passed!")
    except AssertionError as e:
        print(f"Specific test failed: {e}")

    print("\nRunning property-based tests...")
    pytest.main([__file__, "-v"])