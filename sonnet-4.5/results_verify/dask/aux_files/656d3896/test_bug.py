import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Test the specific example from the bug report
divisions = (
    pd.Timestamp('2000-01-01 00:00:00'),
    pd.Timestamp('2000-01-01 01:00:00')
)
rule = 'D'
closed = 'left'
label = 'left'

print("Testing the bug report example:")
print(f"Input divisions: {divisions}")
print(f"Rule: {rule}, closed: {closed}, label: {label}")
print()

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")
print(f"outdivs[0]: {outdivs[0]}")
print(f"outdivs[1]: {outdivs[1]}")
print(f"Are they equal? {outdivs[0] == outdivs[1]}")
print()

# Check if outdivs is monotonically increasing
is_monotonic = True
for i in range(len(outdivs) - 1):
    if outdivs[i] >= outdivs[i+1]:
        is_monotonic = False
        print(f"NOT MONOTONIC: outdivs[{i}] = {outdivs[i]} >= outdivs[{i+1}] = {outdivs[i+1]}")

if is_monotonic:
    print("✓ outdivs is monotonically increasing")
else:
    print("✗ outdivs is NOT monotonically increasing")

print()
print("Testing with the same divisions:")
print(f"len(outdivs): {len(outdivs)}")
print(f"len(divisions): {len(divisions)}")