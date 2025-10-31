#!/usr/bin/env python3

import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Let's trace through the problematic execution
start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2021-12-31')
divisions = pd.date_range(start, end, periods=88)

print("Initial divisions info:")
print(f"  First division: {divisions[0]}")
print(f"  Last division: {divisions[-1]}")
print(f"  Total divisions: {len(divisions)}")

# Trace through the function logic
rule = pd.tseries.frequencies.to_offset('10D')
closed = 'right'
label = 'left'
g = pd.Grouper(freq=rule, how="count", closed=closed, label=label)

divs = pd.Series(range(len(divisions)), index=divisions)
temp = divs.resample('10D', closed='right', label='left').count()
tempdivs = temp.loc[temp > 0].index

print(f"\ntemp series info:")
print(f"  Length of temp: {len(temp)}")
print(f"  Length of tempdivs (filtered): {len(tempdivs)}")
print(f"  Last 5 values in temp.index: {temp.index[-5:]}")
print(f"  Last value in temp.index: {temp.index[-1]}")

res = pd.offsets.Nano() if isinstance(rule, pd.offsets.Tick) else pd.offsets.Day()
print(f"\nres offset: {res}")

# Since closed='right' and label='left'
if g.closed == "right":
    newdivs = tempdivs + res
else:
    newdivs = tempdivs

if g.label == "right":
    outdivs = tempdivs + rule
else:
    outdivs = tempdivs

newdivs = list(newdivs)
outdivs = list(outdivs)

print(f"\nBefore adjustment:")
print(f"  len(newdivs): {len(newdivs)}")
print(f"  len(outdivs): {len(outdivs)}")
print(f"  newdivs[-1]: {newdivs[-1]}")
print(f"  outdivs[-1]: {outdivs[-1]}")
print(f"  divisions[-1]: {divisions[-1]}")

# Adjustment logic
print("\nAdjustment checks:")
print(f"  newdivs[0] < divisions[0]: {newdivs[0] < divisions[0]}")
if newdivs[0] < divisions[0]:
    print(f"    Setting newdivs[0] from {newdivs[0]} to {divisions[0]}")
    newdivs[0] = divisions[0]

print(f"  newdivs[-1] < divisions[-1]: {newdivs[-1] < divisions[-1]} ({newdivs[-1]} < {divisions[-1]})")
if newdivs[-1] < divisions[-1]:
    print(f"    len(newdivs) < len(divs): {len(newdivs) < len(divs)} ({len(newdivs)} < {len(divs)})")

    if len(newdivs) < len(divs):
        print("    Using append strategy")
        print(f"    Appending to newdivs: {divisions[-1] + res}")
        newdivs.append(divisions[-1] + res)

        print(f"    outdivs[-1] > divisions[-1]: {outdivs[-1] > divisions[-1]} ({outdivs[-1]} > {divisions[-1]})")
        print(f"    outdivs[-1] < divisions[-1]: {outdivs[-1] < divisions[-1]} ({outdivs[-1]} < {divisions[-1]})")

        if outdivs[-1] > divisions[-1]:
            print(f"    Would append to outdivs: {outdivs[-1]}")
            outdivs.append(outdivs[-1])
        elif outdivs[-1] < divisions[-1]:
            print(f"    Would append to outdivs: {temp.index[-1]}")
            print(f"    ISSUE: outdivs[-1] = {outdivs[-1]}, temp.index[-1] = {temp.index[-1]}")
            print(f"    ARE THEY EQUAL? {outdivs[-1] == temp.index[-1]}")
            outdivs.append(temp.index[-1])
    else:
        print("    Using setitem strategy")
        # This would use __setitem__

print("\nFinal state:")
print(f"  len(newdivs): {len(newdivs)}")
print(f"  len(outdivs): {len(outdivs)}")
print(f"  outdivs[-2:]: {outdivs[-2:]}")

# Check for duplicate
for i in range(len(outdivs) - 1):
    if outdivs[i] == outdivs[i + 1]:
        print(f"\nDUPLICATE FOUND at index {i}: {outdivs[i]}")
        print(f"  This means outdivs[{i}] == outdivs[{i+1}]")