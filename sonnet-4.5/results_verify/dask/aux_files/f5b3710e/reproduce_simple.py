import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

divisions = (
    pd.Timestamp('2000-01-01 00:00:00'),
    pd.Timestamp('2000-01-02 00:00:00'),
    pd.Timestamp('2000-01-03 00:00:00')
)
rule = '2D'
closed = 'right'
label = 'right'

newdivs, outdivs = _resample_bin_and_out_divs(divisions, rule, closed, label)

print(f"newdivs length: {len(newdivs)}")
print(f"outdivs length: {len(outdivs)}")
print(f"newdivs: {newdivs}")
print(f"outdivs: {outdivs}")

print("\n=== Analysis ===")
print(f"Length mismatch: {len(newdivs) != len(outdivs)}")
print(f"Difference: {len(newdivs) - len(outdivs)}")