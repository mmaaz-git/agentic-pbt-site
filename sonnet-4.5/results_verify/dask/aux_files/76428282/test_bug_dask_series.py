import pandas as pd
import dask.dataframe as dd
import traceback

# Test the actual bug with dask series
try:
    print("Creating test data...")
    index = pd.date_range('2000-01-01 00:00:00', periods=3, freq='6h')
    series = pd.Series(range(len(index)), index=index)
    dask_series = dd.from_pandas(series, npartitions=2)

    print("Performing resample operation...")
    result = dask_series.resample('12h', closed='right', label='right').count()

    print("Computing result...")
    computed = result.compute()

    print("Success! Result:")
    print(computed)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()