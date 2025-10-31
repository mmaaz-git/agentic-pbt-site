import pandas as pd
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag

# Test 1: Simple reproduction
print("=" * 60)
print("Test 1: Simple reproduction")
print("=" * 60)

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
ddf = from_pandas(df, npartitions=2)

bag = to_bag(ddf, format='frame')
result = bag.compute()

print(f"Expected: {ddf.npartitions} DataFrame objects")
print(f"Got: {len(result)} items")
print(f"Result type: {[type(item).__name__ for item in result]}")
print(f"Result: {result}")
print()

# Test 2: Check what we actually get
print("=" * 60)
print("Test 2: Check actual partition content")
print("=" * 60)

# Let's see what the actual partitions look like
for i in range(ddf.npartitions):
    partition = ddf.get_partition(i).compute()
    print(f"Partition {i}: {type(partition).__name__}")
    print(partition)
    print()

# Test 3: Test with format='tuple' for comparison
print("=" * 60)
print("Test 3: Compare with format='tuple'")
print("=" * 60)

bag_tuple = to_bag(ddf, format='tuple')
result_tuple = bag_tuple.compute()
print(f"format='tuple' result length: {len(result_tuple)}")
print(f"format='tuple' first few items: {result_tuple[:3]}")
print()

# Test 4: Test with format='dict' for comparison
print("=" * 60)
print("Test 4: Compare with format='dict'")
print("=" * 60)

bag_dict = to_bag(ddf, format='dict')
result_dict = bag_dict.compute()
print(f"format='dict' result length: {len(result_dict)}")
print(f"format='dict' first few items: {result_dict[:3]}")