import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def divisions_strategy(draw):
    n_divs = draw(st.integers(min_value=2, max_value=20))
    start = draw(st.datetimes(
        min_value=pd.Timestamp("2000-01-01"),
        max_value=pd.Timestamp("2020-01-01")
    ))
    freq = draw(st.sampled_from(['1h', '1D', '1min', '30min', '1W']))
    divisions = pd.date_range(start=start, periods=n_divs, freq=freq)
    return tuple(divisions)


@given(
    divisions=divisions_strategy(),
    rule=st.sampled_from(['1h', '2h', '1D', '2D', '1W', '30min', '15min']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=1000)
def test_resample_bin_and_out_divs_monotonic(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] <= outdivs[i+1], f"outdivs not monotonic: {outdivs[i]} > {outdivs[i+1]}"


if __name__ == "__main__":
    # Run the test
    test_resample_bin_and_out_divs_monotonic()