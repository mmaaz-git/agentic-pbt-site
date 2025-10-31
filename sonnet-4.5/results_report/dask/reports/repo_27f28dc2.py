import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Test case from the bug report
divisions = (pd.Timestamp('2001-02-04 00:00:00'), pd.Timestamp('2001-02-04 01:00:00'))
rule = '1W'
closed = 'right'
label = 'right'

print(f"Testing with:")
print(f"  divisions = {divisions}")
print(f"  rule = '{rule}'")
print(f"  closed = '{closed}'")
print(f"  label = '{label}'")
print()

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

print(f"Result:")
print(f"  newdivs: {newdivs}")
print(f"  outdivs: {outdivs}")
print()

# Check if outdivs is monotonic
is_monotonic = all(outdivs[i] <= outdivs[i+1] for i in range(len(outdivs)-1))
print(f"Monotonicity check:")
print(f"  Are outdivs monotonic? {is_monotonic}")

if not is_monotonic:
    print(f"\nERROR: Non-monotonic divisions detected!")
    for i in range(len(outdivs)-1):
        if outdivs[i] > outdivs[i+1]:
            print(f"  outdivs[{i}] = {outdivs[i]} > outdivs[{i+1}] = {outdivs[i+1]}")