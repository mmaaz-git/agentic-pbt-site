import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=2)

print("Input divisions:")
for i, d in enumerate(divisions):
    print(f"  divisions[{i}]: {d}")

newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'h', closed='right', label='left')

print("\nOutput divisions:")
print(f"  First newdiv: {newdivs[0]}")
print(f"  Last newdiv: {newdivs[-1]}")
print(f"  First outdiv: {outdivs[0]}")
print(f"  Last outdiv: {outdivs[-1]}")

print(f"\nInput divisions[0]: {divisions[0]}")
print(f"Output outdivs[0]: {outdivs[0]}")

if outdivs[0] < divisions[0]:
    print(f"\nERROR: outdivs[0] is before divisions[0]")
    print(f"  {outdivs[0]} < {divisions[0]}")
    print(f"  Difference: {divisions[0] - outdivs[0]}")
else:
    print(f"\nNo issue found: outdivs[0] >= divisions[0]")