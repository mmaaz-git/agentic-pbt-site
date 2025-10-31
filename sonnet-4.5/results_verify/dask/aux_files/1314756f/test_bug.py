import pandas as pd
import dask.dataframe as dd

# Test simple reproduction case
s = pd.Series(['0', '1', '2'])
ds = dd.from_pandas(s, npartitions=2)

result_pandas = pd.to_numeric(s)
result_dask = dd.to_numeric(ds).compute()

print(f'Pandas dtype: {result_pandas.dtype}')
print(f'Dask dtype: {result_dask.dtype}')

print(f'Pandas values: {result_pandas.values}')
print(f'Dask values: {result_dask.values}')

# Check if dtypes match
if result_dask.dtype == result_pandas.dtype:
    print("DTYPES MATCH")
else:
    print("DTYPES DO NOT MATCH - ASSERTION WOULD FAIL")

# Test with floats
s2 = pd.Series(['0.5', '1.5', '2.5'])
ds2 = dd.from_pandas(s2, npartitions=2)

result_pandas2 = pd.to_numeric(s2)
result_dask2 = dd.to_numeric(ds2).compute()

print(f'\nFloat test - Pandas dtype: {result_pandas2.dtype}')
print(f'Float test - Dask dtype: {result_dask2.dtype}')

# Test the hypothesis example
from hypothesis import given, settings, strategies as st

@given(st.lists(st.integers(min_value=-1000, max_value=1000).map(str), min_size=1, max_size=100))
@settings(max_examples=10)
def test_to_numeric_dtype_matches_pandas(values):
    s = pd.Series(values)
    ds = dd.from_pandas(s, npartitions=2)

    result_pandas = pd.to_numeric(s)
    result_dask = dd.to_numeric(ds).compute()

    assert result_dask.dtype == result_pandas.dtype, \
        f"Expected dtype {result_pandas.dtype}, got {result_dask.dtype}"

print("\nRunning property test...")
try:
    test_to_numeric_dtype_matches_pandas()
    print("All tests passed")
except AssertionError as e:
    print(f"AssertionError: {e}")