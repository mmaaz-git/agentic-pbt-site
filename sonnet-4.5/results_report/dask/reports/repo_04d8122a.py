import pandas as pd
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag

# Create a simple DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("Original DataFrame:")
print(df)
print()

# Convert to Dask DataFrame with 2 partitions
ddf = from_pandas(df, npartitions=2)
print(f"Dask DataFrame with {ddf.npartitions} partitions")
print()

# Try to convert to bag with format='frame'
print("Attempting to_bag(ddf, format='frame')...")
bag = to_bag(ddf, format='frame')
result = bag.compute()

print(f"Expected: {ddf.npartitions} DataFrame objects")
print(f"Got: {len(result)} items of type {type(result[0]) if result else 'None'}")
print(f"Result: {result}")
print()

# Show what each partition should look like
print("What we expected (DataFrame partitions):")
for i in range(ddf.npartitions):
    partition = ddf.get_partition(i).compute()
    print(f"Partition {i}:")
    print(partition)
    print()