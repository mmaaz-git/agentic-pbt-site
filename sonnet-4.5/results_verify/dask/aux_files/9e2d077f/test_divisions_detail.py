import pandas as pd
import dask.dataframe as dd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Let's check the divisions for both cases
# Case 1: Works (48h in 4 partitions)
series1 = pd.Series(
    list(range(48)),
    index=pd.date_range('2000-01-01 00:00:00', periods=48, freq='h')
)
ds1 = dd.from_pandas(series1, npartitions=4)
print("Case 1: 48h in 4 partitions (WORKS)")
print(f"Original divisions: {ds1.divisions}")
print(f"Original npartitions: {ds1.npartitions}")

resample_divs1 = _resample_bin_and_out_divs(ds1.divisions, 'D', closed=None, label=None)
print(f"Resample bin divisions: {resample_divs1[0]}")
print(f"Resample output divisions: {resample_divs1[1]}")
print(f"Number of bin partitions: {len(resample_divs1[0]) - 1}")
print(f"Number of output partitions: {len(resample_divs1[1]) - 1}")

print("\n" + "="*60)

# Case 2: Fails (10h in 5 partitions)
series2 = pd.Series(
    list(range(10)),
    index=pd.date_range('2000-01-01 00:00:00', periods=10, freq='h')
)
ds2 = dd.from_pandas(series2, npartitions=5)
print("Case 2: 10h in 5 partitions (FAILS)")
print(f"Original divisions: {ds2.divisions}")
print(f"Original npartitions: {ds2.npartitions}")

resample_divs2 = _resample_bin_and_out_divs(ds2.divisions, 'D', closed=None, label=None)
print(f"Resample bin divisions: {resample_divs2[0]}")
print(f"Resample output divisions: {resample_divs2[1]}")
print(f"Number of bin partitions: {len(resample_divs2[0]) - 1}")
print(f"Number of output partitions: {len(resample_divs2[1]) - 1}")