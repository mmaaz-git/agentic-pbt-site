import dask.dataframe as dd
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({'x': range(100)})
dask_df = dd.from_pandas(df, npartitions=5)

print("="*60)
print("HEAD() DOCUMENTATION:")
print("="*60)
print(dask_df.head.__doc__)

print("\n" + "="*60)
print("TAIL() DOCUMENTATION:")
print("="*60)
print(dask_df.tail.__doc__)

print("\n" + "="*60)
print("Checking help() output:")
print("="*60)
print("\nHelp for head():")
help(dask_df.head)
print("\nHelp for tail():")
help(dask_df.tail)