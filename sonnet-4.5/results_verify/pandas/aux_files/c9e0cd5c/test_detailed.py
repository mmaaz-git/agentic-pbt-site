import pandas as pd
import numpy as np
from datetime import datetime

# Test with different data types
df = pd.DataFrame({
    'int_col': [1, 2, 3],
    'float_col': [1.5, 2.5, 3.5],
    'str_col': ['a', 'b', 'c'],
    'date_col': [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
    'bool_col': [True, False, True]
})

print("Testing DataFrame:")
print(df)
print("\nData types:")
print(df.dtypes)
print()

split_result = df.to_dict(orient='split')
tight_result = df.to_dict(orient='tight')

print("Split result data:")
print(split_result['data'])
print("\nTight result data:")
print(tight_result['data'])
print("\nAre they equal?", split_result['data'] == tight_result['data'])

# Let's also check if they are functionally equivalent
print("\nChecking element by element:")
for i, (s_row, t_row) in enumerate(zip(split_result['data'], tight_result['data'])):
    print(f"Row {i}:")
    for j, (s_val, t_val) in enumerate(zip(s_row, t_row)):
        match = s_val == t_val
        print(f"  Col {j}: split={s_val!r} tight={t_val!r} equal={match}")

# Let's inspect the source code behavior more directly
print("\n\nChecking the actual code path:")

# Simulate what the code does
are_all_object_dtype_cols = False  # Since we have mixed types
object_dtype_indices = [2, 3, 4]  # str_col, date_col, bool_col are object dtype

print(f"are_all_object_dtype_cols: {are_all_object_dtype_cols}")
print(f"object_dtype_indices: {object_dtype_indices}")

# This is what _create_data_for_split_and_tight_to_dict would do
from pandas._libs.lib import maybe_box_native

# Optimized path (used by split)
data_optimized = [list(t) for t in df.itertuples(index=False, name=None)]
if object_dtype_indices:
    for row in data_optimized:
        for i in object_dtype_indices:
            row[i] = maybe_box_native(row[i])

print("\nOptimized data (what split uses):")
print(data_optimized)

# Unoptimized path (what tight currently recomputes)
data_unoptimized = [
    list(map(maybe_box_native, t))
    for t in df.itertuples(index=False, name=None)
]

print("\nUnoptimized data (what tight recomputes):")
print(data_unoptimized)

print("\nAre they equal?", data_optimized == data_unoptimized)