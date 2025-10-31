import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Create divisions with 2 points
start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=2)

# Call the function with the failing parameters
newdivs, outdivs = _resample_bin_and_out_divs(divisions, 'h', closed='right', label='left')

print(f"Input divisions[0]: {divisions[0]}")
print(f"Input divisions[-1]: {divisions[-1]}")
print()
print(f"Output newdivs[0]: {newdivs[0]}")
print(f"Output outdivs[0]: {outdivs[0]}")
print()

# Check if outdivs extends before the input range
if outdivs[0] < divisions[0]:
    print(f"ERROR: outdivs[0] is before divisions[0]")
    print(f"  outdivs[0] = {outdivs[0]}")
    print(f"  divisions[0] = {divisions[0]}")
    print(f"  Difference: {divisions[0] - outdivs[0]}")
else:
    print("OK: outdivs[0] is within the input range")