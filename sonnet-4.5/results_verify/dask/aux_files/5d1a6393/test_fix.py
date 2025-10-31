import pandas as pd
import numpy as np
from datetime import datetime

def _resample_bin_and_out_divs_fixed(divisions, rule, closed="left", label="left"):
    """Fixed version of the function with the proposed patch"""
    rule = pd.tseries.frequencies.to_offset(rule)
    g = pd.Grouper(freq=rule, how="count", closed=closed, label=label)

    # Determine bins to apply `how` to. Disregard labeling scheme.
    divs = pd.Series(range(len(divisions)), index=divisions)
    temp = divs.resample(rule, closed=closed, label="left").count()
    tempdivs = temp.loc[temp > 0].index

    # Cleanup closed == 'right' and label == 'right'
    res = pd.offsets.Nano() if isinstance(rule, pd.offsets.Tick) else pd.offsets.Day()
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

    # Adjust ends
    if newdivs[0] < divisions[0]:
        newdivs[0] = divisions[0]
    if newdivs[-1] < divisions[-1]:
        if len(newdivs) < len(divs):
            setter = lambda a, val: a.append(val)
        else:
            setter = lambda a, val: a.__setitem__(-1, val)
        setter(newdivs, divisions[-1] + res)
        if outdivs[-1] > divisions[-1]:
            setter(outdivs, outdivs[-1])
        elif outdivs[-1] < divisions[-1]:
            # FIXED: When label='right', shift the value appropriately
            if g.label == "right":
                setter(outdivs, temp.index[-1] + rule)
            else:
                setter(outdivs, temp.index[-1])

    return tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs))

# Test the fix with the failing case
print("=== TESTING FIXED VERSION ===")
divisions = pd.DatetimeIndex(
    ['2020-02-29 00:00:00', '2020-02-29 00:01:00'],
    dtype='datetime64[ns]',
    freq='min'
)
rule = '1M'
closed = 'right'
label = 'right'

newdivs, outdivs = _resample_bin_and_out_divs_fixed(divisions, rule, closed, label)

print(f"divisions: {divisions}")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")
print(f"outdivs sorted? {list(outdivs) == sorted(outdivs)}")
print()

# Test with more cases
test_cases = [
    (pd.DatetimeIndex(['2020-02-29 00:00:00', '2020-02-29 00:01:00'], freq='min'), '1M', 'right', 'right'),
    (pd.DatetimeIndex(['2021-02-07 00:00:00', '2021-02-07 00:01:00'], freq='min'), '1W', 'right', 'right'),
    (pd.DatetimeIndex(['2020-01-31 00:00:00', '2020-01-31 00:01:00'], freq='min'), '1M', 'right', 'right'),
]

print("=== TESTING MULTIPLE CASES ===")
for divs, r, c, l in test_cases:
    newdivs, outdivs = _resample_bin_and_out_divs_fixed(divs, r, c, l)
    is_sorted = list(outdivs) == sorted(outdivs)
    print(f"Rule={r}, closed={c}, label={l}: Sorted={is_sorted}, outdivs={outdivs}")