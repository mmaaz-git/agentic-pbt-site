#!/usr/bin/env python3
"""Analyze the specific code section that causes the bug"""

import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Test case that triggers the bug
start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=7)

print("Tracing through the function logic...")
print(f"Input divisions: {list(divisions)}")

# Simulate what happens in the function
rule = pd.tseries.frequencies.to_offset('W')
g = pd.Grouper(freq=rule, how="count", closed='left', label='left')

# Determine bins to apply `how` to. Disregard labeling scheme.
divs = pd.Series(range(len(divisions)), index=divisions)
temp = divs.resample(rule, closed='left', label="left").count()
tempdivs = temp.loc[temp > 0].index

print(f"\ntempdivs: {list(tempdivs)}")
print(f"temp.index[-1]: {temp.index[-1]}")

# Cleanup closed == 'left' and label == 'left'
res = pd.offsets.Nano() if isinstance(rule, pd.offsets.Tick) else pd.offsets.Day()
print(f"res (offset): {res}")

if g.closed == 'right':
    newdivs = tempdivs + res
else:
    newdivs = tempdivs

if g.label == 'right':
    outdivs = tempdivs + rule
else:
    outdivs = tempdivs

newdivs = list(newdivs)
outdivs = list(outdivs)

print(f"\nBefore adjustment:")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")

# Adjust ends - this is where the bug occurs
print(f"\nAdjusting ends...")
print(f"newdivs[0] ({newdivs[0]}) < divisions[0] ({divisions[0]}): {newdivs[0] < divisions[0]}")

if newdivs[0] < divisions[0]:
    newdivs[0] = divisions[0]

print(f"newdivs[-1] ({newdivs[-1]}) < divisions[-1] ({divisions[-1]}): {newdivs[-1] < divisions[-1]}")

if newdivs[-1] < divisions[-1]:
    print(f"  len(newdivs) ({len(newdivs)}) < len(divs) ({len(divs)}): {len(newdivs) < len(divs)}")
    if len(newdivs) < len(divs):
        setter = lambda a, val: a.append(val)
        print("  Using append setter")
    else:
        setter = lambda a, val: a.__setitem__(-1, val)
        print("  Using setitem setter")

    print(f"  Setting newdivs: divisions[-1] + res = {divisions[-1]} + {res} = {divisions[-1] + res}")
    setter(newdivs, divisions[-1] + res)

    print(f"  outdivs[-1] ({outdivs[-1]}) > divisions[-1] ({divisions[-1]}): {outdivs[-1] > divisions[-1]}")

    if outdivs[-1] > divisions[-1]:
        print(f"  BUG: About to setter(outdivs, outdivs[-1]) which will append/set {outdivs[-1]} to outdivs!")
        print(f"  This creates a duplicate because outdivs already ends with {outdivs[-1]}")
        setter(outdivs, outdivs[-1])
    elif outdivs[-1] < divisions[-1]:
        print(f"  Setting outdivs to temp.index[-1] = {temp.index[-1]}")
        setter(outdivs, temp.index[-1])

print(f"\nAfter adjustment:")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")

# Check for the bug
print("\n" + "="*60)
print("BUG CHECK:")
if len(outdivs) > 1 and outdivs[-1] == outdivs[-2]:
    print(f"✗ BUG CONFIRMED: Last two elements of outdivs are identical: {outdivs[-1]}")
else:
    print("✓ No duplicate at end of outdivs")