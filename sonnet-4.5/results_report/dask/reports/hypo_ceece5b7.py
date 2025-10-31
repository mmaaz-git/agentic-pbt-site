import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@st.composite
def timestamp_list_strategy(draw):
    size = draw(st.integers(min_value=2, max_value=20))
    start = draw(st.datetimes(
        min_value=pd.Timestamp('2000-01-01'),
        max_value=pd.Timestamp('2020-01-01')
    ))
    freq_hours = draw(st.integers(min_value=1, max_value=24*7))
    timestamps = pd.date_range(start=start, periods=size, freq=f'{freq_hours}h')
    return timestamps.tolist()

@given(
    divisions=timestamp_list_strategy(),
    rule=st.sampled_from(['1h', '2h', '6h', '12h', '1D', '2D']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=500, deadline=None)
def test_no_duplicate_divisions(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] < outdivs[i+1], \
            f"Divisions not strictly increasing: outdivs[{i}]={outdivs[i]}, outdivs[{i+1}]={outdivs[i+1]}"

if __name__ == "__main__":
    # Run the test to find failing cases
    test_no_duplicate_divisions()