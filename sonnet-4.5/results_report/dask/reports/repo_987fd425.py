import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

segments = []
for start_str in ['2020-01-01', '2020-02-01']:
    start = pd.Timestamp(start_str)
    end = start + pd.Timedelta(days=7)
    segment = pd.date_range(start, end, periods=9)
    segments.append(segment)

divisions = segments[0].union(segments[1])

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'D', closed='right', label='right')

print(f"len(newdivs) = {len(newdivs)}")
print(f"len(outdivs) = {len(outdivs)}")
print(f"Length mismatch: {len(newdivs) != len(outdivs)}")
print(f"\nnewdivs: {newdivs}")
print(f"\noutdivs: {outdivs}")