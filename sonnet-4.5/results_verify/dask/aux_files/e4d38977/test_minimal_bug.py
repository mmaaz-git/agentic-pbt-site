import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Reproduce with the minimal example from the bug report
divisions = pd.date_range('2020-01-01', periods=2, freq='D')
newdivs, outdivs = _resample_bin_and_out_divs(divisions, '2D', closed='left', label='left')

print("divisions:", divisions)
print("newdivs:", newdivs)
print("outdivs:", outdivs)
print("Has duplicates in outdivs:", len(outdivs) != len(set(outdivs)))

# Check if duplicates exist
if len(outdivs) != len(set(outdivs)):
    print("CONFIRMED: outdivs contains duplicate timestamps!")
    for i in range(len(outdivs)-1):
        if outdivs[i] == outdivs[i+1]:
            print(f"  Duplicate found at index {i} and {i+1}: {outdivs[i]}")