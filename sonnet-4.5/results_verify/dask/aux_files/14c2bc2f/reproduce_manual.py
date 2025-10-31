import dask.dataframe as dd
import pandas as pd

df = pd.DataFrame({'x': [0, 0], 'y': [0, 0]})

pandas_result = df.reset_index(drop=True)
print("Pandas result:")
print(pandas_result)
print(f"Index: {pandas_result.index.tolist()}")

ddf = dd.from_pandas(df, npartitions=2)
dask_result = ddf.reset_index(drop=True).compute()
print("\nDask result:")
print(dask_result)
print(f"Index: {dask_result.index.tolist()}")

print("\n" + "="*50)
print("Testing with more data to see the pattern:")

df2 = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': [10, 20, 30, 40, 50, 60]})
pandas_result2 = df2.reset_index(drop=True)
print("\nPandas result (6 rows):")
print(pandas_result2)
print(f"Index: {pandas_result2.index.tolist()}")

ddf2 = dd.from_pandas(df2, npartitions=3)  # 3 partitions
dask_result2 = ddf2.reset_index(drop=True).compute()
print("\nDask result (6 rows, 3 partitions):")
print(dask_result2)
print(f"Index: {dask_result2.index.tolist()}")