#!/usr/bin/env python3
"""Run the Hypothesis test from the bug report"""

import pandas as pd
from hypothesis import given, strategies as st, settings, example
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

failures = []

@given(
    divisions=timestamp_list_strategy(),
    rule=st.sampled_from(['1h', '2h', '6h', '12h', '1D', '2D', '1W']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@example(
    divisions=[pd.Timestamp('2000-12-17 00:00:00'), pd.Timestamp('2000-12-17 01:00:00')],
    rule='1W',
    closed='right',
    label='right'
)
@settings(max_examples=100, deadline=None)
def test_resample_bin_and_out_divs_monotonic(divisions, rule, closed, label):
    newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

    for i in range(len(outdivs) - 1):
        if outdivs[i] > outdivs[i+1]:
            failure_info = {
                'divisions': divisions,
                'rule': rule,
                'closed': closed,
                'label': label,
                'outdivs': outdivs,
                'failing_indices': (i, i+1),
                'failing_values': (outdivs[i], outdivs[i+1])
            }
            failures.append(failure_info)
            assert False, \
                f"outdivs not monotonic: outdivs[{i}]={outdivs[i]} > outdivs[{i+1}]={outdivs[i+1]}"

print("Running Hypothesis test...")
try:
    test_resample_bin_and_out_divs_monotonic()
    print("\nNo failures detected in random testing!")
except AssertionError as e:
    print(f"\n❌ Test failed: {e}")
    if failures:
        print(f"\nFound {len(failures)} failure case(s):")
        for idx, failure in enumerate(failures[:5], 1):  # Show first 5 failures
            print(f"\n{idx}. divisions={failure['divisions'][:2]}...")
            print(f"   rule={failure['rule']}, closed={failure['closed']}, label={failure['label']}")
            print(f"   Problem: outdivs[{failure['failing_indices'][0]}]={failure['failing_values'][0]} > "
                  f"outdivs[{failure['failing_indices'][1]}]={failure['failing_values'][1]}")