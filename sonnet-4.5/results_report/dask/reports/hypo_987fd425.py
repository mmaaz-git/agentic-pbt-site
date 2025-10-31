import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

@given(
    st.integers(min_value=3, max_value=50),
    st.sampled_from(['h', 'D', '2h', '6h', '12h']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=1000)
def test_resample_divisions_with_gaps(n_segments, rule, closed, label):
    segments = []
    start_dates = ['2020-01-01', '2020-02-01']

    for start_str in start_dates:
        start = pd.Timestamp(start_str)
        end = start + pd.Timedelta(days=7)
        segment = pd.date_range(start, end, periods=n_segments // 2)
        segments.append(segment)

    divisions = segments[0].union(segments[1])

    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

    assert len(newdivs) == len(outdivs), f"Length mismatch: newdivs={len(newdivs)}, outdivs={len(outdivs)}"

if __name__ == "__main__":
    test_resample_divisions_with_gaps()