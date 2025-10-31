#!/usr/bin/env python3
"""Test to reproduce the reported bug in dask.dataframe.tseries.resample._resample_bin_and_out_divs"""

import pandas as pd
from hypothesis import given, strategies as st, settings, seed
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# First, let's reproduce the specific failing case
print("=" * 60)
print("REPRODUCING SPECIFIC FAILING CASE")
print("=" * 60)

divisions = [pd.Timestamp('2000-01-01 00:00:00'), pd.Timestamp('2000-01-01 01:00:00')]
newdivs, outdivs = _resample_bin_and_out_divs(divisions, '2h', closed='left', label='left')

print(f"Input divisions: {divisions}")
print(f"Rule: 2h, closed: left, label: left")
print(f"Output newdivs: {newdivs}")
print(f"Output outdivs: {outdivs}")

# Check for duplicates
has_duplicates = False
for i in range(len(outdivs) - 1):
    if outdivs[i] == outdivs[i+1]:
        print(f"DUPLICATE FOUND: outdivs[{i}] == outdivs[{i+1}] == {outdivs[i]}")
        has_duplicates = True
    if outdivs[i] >= outdivs[i+1]:
        print(f"NOT STRICTLY INCREASING: outdivs[{i}]={outdivs[i]} >= outdivs[{i+1}]={outdivs[i+1]}")

if not has_duplicates:
    print("No duplicates found in outdivs")

# Now let's run the hypothesis test
print("\n" + "=" * 60)
print("RUNNING HYPOTHESIS TEST")
print("=" * 60)

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

failed_cases = []

@given(
    divisions=timestamp_list_strategy(),
    rule=st.sampled_from(['1h', '2h', '6h', '12h', '1D', '2D']),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=100, deadline=None)
@seed(12345)  # For reproducibility
def test_no_duplicate_divisions(divisions, rule, closed, label):
    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

        for i in range(len(outdivs) - 1):
            if outdivs[i] >= outdivs[i+1]:
                failed_cases.append({
                    'divisions': divisions,
                    'rule': rule,
                    'closed': closed,
                    'label': label,
                    'outdivs': outdivs,
                    'issue': f"outdivs[{i}]={outdivs[i]} >= outdivs[{i+1}]={outdivs[i+1]}"
                })
                assert False, f"Divisions not strictly increasing: outdivs[{i}]={outdivs[i]}, outdivs[{i+1}]={outdivs[i+1]}"
    except Exception as e:
        if "Divisions not strictly increasing" not in str(e):
            raise

# Run the test
try:
    test_no_duplicate_divisions()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Print failed cases summary
if failed_cases:
    print(f"\nFound {len(failed_cases)} failing cases. First few examples:")
    for i, case in enumerate(failed_cases[:3]):
        print(f"\nFailed case {i+1}:")
        print(f"  Divisions: {case['divisions'][:3]}... ({len(case['divisions'])} total)")
        print(f"  Rule: {case['rule']}, Closed: {case['closed']}, Label: {case['label']}")
        print(f"  Issue: {case['issue']}")
        print(f"  Outdivs: {case['outdivs']}")