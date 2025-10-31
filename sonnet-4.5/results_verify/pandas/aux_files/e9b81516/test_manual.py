import pandas as pd

df = pd.DataFrame({'A': pd.Series([], dtype='int64'), 'B': pd.Series([], dtype='int64')})

print(f"Original dtypes:\n{df.dtypes}")
print(f"\nOriginal DataFrame:\n{df}")

result = df.T.T

print(f"\nAfter df.T.T dtypes:\n{result.dtypes}")
print(f"\nResulting DataFrame:\n{result}")

print("\nTesting with non-empty DataFrame:")
df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"\nOriginal dtypes:\n{df2.dtypes}")
result2 = df2.T.T
print(f"\nAfter df.T.T dtypes:\n{result2.dtypes}")