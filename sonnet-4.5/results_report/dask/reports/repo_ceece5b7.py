import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Minimal test case showing the bug
divisions = [pd.Timestamp('2000-01-01 00:00:00'), pd.Timestamp('2000-01-01 01:00:00')]
rule = '2h'
closed = 'left'
label = 'left'

print("Input:")
print(f"  divisions: {divisions}")
print(f"  rule: '{rule}'")
print(f"  closed: '{closed}'")
print(f"  label: '{label}'")
print()

# Call the function
newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed=closed, label=label)

print("Output:")
print(f"  newdivs: {newdivs}")
print(f"  outdivs: {outdivs}")
print()

# Check for duplicates in outdivs
print("Checking for duplicates in outdivs:")
for i in range(len(outdivs) - 1):
    if outdivs[i] >= outdivs[i+1]:
        print(f"  ERROR: outdivs[{i}] ({outdivs[i]}) >= outdivs[{i+1}] ({outdivs[i+1]})")
        print(f"  Divisions are not strictly monotonic increasing!")
    else:
        print(f"  OK: outdivs[{i}] < outdivs[{i+1}]")

if len(outdivs) > 1 and outdivs[0] == outdivs[-1]:
    print()
    print("CRITICAL BUG: All division values are identical!")
    print("This violates Dask's fundamental requirement that divisions be strictly monotonic.")