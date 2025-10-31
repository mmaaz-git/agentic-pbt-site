#!/usr/bin/env python3

import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# First, let's run the specific reproduction case
print("=" * 80)
print("REPRODUCING THE SPECIFIC BUG CASE")
print("=" * 80)

start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2021-12-31')
divisions = pd.date_range(start, end, periods=88)

newdivs, outdivs = _resample_bin_and_out_divs(divisions, '10D', closed='right', label='left')

print(f"Number of newdivs: {len(newdivs)}")
print(f"Number of outdivs: {len(outdivs)}")
print(f"outdivs[-5:] = {outdivs[-5:]}")

duplicate_found = False
for i in range(len(outdivs) - 1):
    if outdivs[i] == outdivs[i + 1]:
        print(f"DUPLICATE at index {i}: {outdivs[i]}")
        duplicate_found = True

if not duplicate_found:
    print("No duplicates found")
else:
    print("\nDuplicate confirmed!")

# Now let's run the property-based test
print("\n" + "=" * 80)
print("RUNNING PROPERTY-BASED TEST")
print("=" * 80)

@given(
    st.integers(min_value=3, max_value=100),
    st.sampled_from(['h', 'D', '2h', '3h', '6h', '12h', '30min', 'W', '2D', '3D', '5D', '7D', '10D']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right'])
)
@settings(max_examples=200, deadline=None)  # Reduced for faster testing
def test_outdivs_no_consecutive_duplicates(n_divisions, rule, closed, label):
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2021-12-31')
    divisions = pd.date_range(start, end, periods=n_divisions)

    try:
        newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)
    except Exception:
        assume(False)

    for i in range(len(outdivs) - 1):
        if outdivs[i] == outdivs[i + 1]:
            print(f"Found duplicate with params: n_divisions={n_divisions}, rule='{rule}', closed='{closed}', label='{label}'")
            print(f"  Duplicate at index {i}: {outdivs[i]}")
            assert False, f"Found consecutive duplicate in outdivs at index {i}: {outdivs[i]}"

print("Running property-based test...")
try:
    test_outdivs_no_consecutive_duplicates()
    print("Property-based test passed")
except AssertionError as e:
    print(f"Property-based test failed: {e}")
except Exception as e:
    print(f"Property-based test error: {e}")