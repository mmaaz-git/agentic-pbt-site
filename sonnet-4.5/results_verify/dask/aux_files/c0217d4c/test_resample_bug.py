#!/usr/bin/env python3

# Test 1: Hypothesis test
from hypothesis import given, strategies as st, settings, assume
import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs


@st.composite
def date_ranges(draw):
    start_year = draw(st.integers(min_value=2000, max_value=2023))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))
    periods = draw(st.integers(min_value=2, max_value=100))
    freq = draw(st.sampled_from(['h', 'D', '30min', '2h', '6h']))
    start = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    dates = pd.date_range(start, periods=periods, freq=freq)
    return dates


@st.composite
def resample_rules(draw):
    return draw(st.sampled_from(['30min', 'h', '2h', 'D', 'W']))


@given(
    dates=date_ranges(),
    rule=resample_rules(),
    closed=st.sampled_from(['left', 'right']),
    label=st.sampled_from(['left', 'right'])
)
@settings(max_examples=50)  # Reduced for quick testing
def test_resample_bin_and_out_divs_returns_sorted_divisions(dates, rule, closed, label):
    divisions = list(dates)
    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception:
        assume(False)

    for i in range(len(outdivs) - 1):
        assert outdivs[i] <= outdivs[i + 1], f"outdivs not sorted: {outdivs[i]} > {outdivs[i + 1]}"


# Test 2: Direct reproduction
print("Running Hypothesis test...")
try:
    test_resample_bin_and_out_divs_returns_sorted_divisions()
    print("Hypothesis test PASSED")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")

print("\n" + "="*80 + "\n")
print("Running direct reproduction with specific failing input...")

dates = pd.date_range('2001-02-03 00:00:00', periods=26, freq='h')
divisions = list(dates)

print(f"Input divisions (first 3): {divisions[:3]}")
print(f"Input divisions (last 3): {divisions[-3:]}")

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'W', closed='right', label='right')

print(f"\nOutput outdivs: {outdivs}")
print(f"Output newdivs: {newdivs}")

print(f"\nChecking if outdivs are sorted:")
for i in range(len(outdivs) - 1):
    if outdivs[i] > outdivs[i + 1]:
        print(f"ERROR: outdivs[{i}] = {outdivs[i]} > outdivs[{i+1}] = {outdivs[i + 1]}")
        print("This violates the requirement that divisions must be sorted!")
        break
else:
    print("outdivs are correctly sorted")

# Verify the specific assertion from the bug report
try:
    assert outdivs[0] <= outdivs[1], f"outdivs[0] ({outdivs[0]}) is not <= outdivs[1] ({outdivs[1]})"
    print("\nAssertion passed: outdivs[0] <= outdivs[1]")
except AssertionError as e:
    print(f"\nAssertion FAILED: {e}")