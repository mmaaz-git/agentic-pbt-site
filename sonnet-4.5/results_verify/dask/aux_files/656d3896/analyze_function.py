import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Let's trace through the function with the failing example
divisions = (
    pd.Timestamp('2000-01-01 00:00:00'),
    pd.Timestamp('2000-01-01 01:00:00')
)
rule = 'D'
closed = 'left'
label = 'left'

print("Tracing through _resample_bin_and_out_divs:")
print(f"Input: divisions={divisions}, rule={rule}, closed={closed}, label={label}")
print()

# Reproducing the function logic step by step
rule_offset = pd.tseries.frequencies.to_offset(rule)
print(f"rule_offset: {rule_offset}")

g = pd.Grouper(freq=rule_offset, how="count", closed=closed, label=label)
print(f"Grouper: freq={rule_offset}, closed={closed}, label={label}")

# Create series to resample
divs = pd.Series(range(len(divisions)), index=divisions)
print(f"\ndivs Series:\n{divs}")

temp = divs.resample(rule, closed=closed, label="left").count()
print(f"\ntemp after resample:\n{temp}")

tempdivs = temp.loc[temp > 0].index
print(f"\ntempdivs (where count > 0): {list(tempdivs)}")

# Determine res
res = pd.offsets.Nano() if isinstance(rule_offset, pd.offsets.Tick) else pd.offsets.Day()
print(f"\nres offset: {res}")

# Calculate newdivs and outdivs
if g.closed == "right":
    newdivs = tempdivs + res
else:
    newdivs = tempdivs

if g.label == "right":
    outdivs = tempdivs + rule_offset
else:
    outdivs = tempdivs

print(f"\nBefore conversion to list:")
print(f"newdivs: {list(newdivs)}")
print(f"outdivs: {list(outdivs)}")

# Convert to list
newdivs = list(newdivs)
outdivs = list(outdivs)

print(f"\nAfter conversion to list:")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")

# Adjustment logic
print(f"\nAdjustment checks:")
print(f"newdivs[0] < divisions[0]? {newdivs[0]} < {divisions[0]} = {newdivs[0] < divisions[0]}")
if newdivs[0] < divisions[0]:
    newdivs[0] = divisions[0]
    print(f"Set newdivs[0] = {divisions[0]}")

print(f"newdivs[-1] < divisions[-1]? {newdivs[-1]} < {divisions[-1]} = {newdivs[-1] < divisions[-1]}")

if newdivs[-1] < divisions[-1]:
    print(f"len(newdivs) < len(divs)? {len(newdivs)} < {len(divs)} = {len(newdivs) < len(divs)}")
    if len(newdivs) < len(divs):
        setter = lambda a, val: a.append(val)
        print("Using append setter")
    else:
        setter = lambda a, val: a.__setitem__(-1, val)
        print("Using setitem(-1) setter")

    print(f"Setting newdivs with divisions[-1] + res = {divisions[-1]} + {res} = {divisions[-1] + res}")
    setter(newdivs, divisions[-1] + res)

    print(f"outdivs[-1] > divisions[-1]? {outdivs[-1]} > {divisions[-1]} = {outdivs[-1] > divisions[-1]}")
    print(f"outdivs[-1] < divisions[-1]? {outdivs[-1]} < {divisions[-1]} = {outdivs[-1] < divisions[-1]}")

    if outdivs[-1] > divisions[-1]:
        print(f"Setting outdivs with outdivs[-1] = {outdivs[-1]}")
        setter(outdivs, outdivs[-1])
    elif outdivs[-1] < divisions[-1]:
        print(f"Setting outdivs with temp.index[-1] = {temp.index[-1]}")
        setter(outdivs, temp.index[-1])

print(f"\nFinal result:")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")