import pandas as pd
from pandas import RangeIndex
import numpy as np

# Create a single-element RangeIndex
ri = RangeIndex(0, 1, 1)
print(f"Original RangeIndex: {ri}")
print(f"Original RangeIndex values: {list(ri)}")
print(f"Original type: {type(ri).__name__}")
print()

# Create a numpy array with the single element
values = np.array([0])
print(f"Values to shallow copy: {values}")
print()

# Call _shallow_copy
result = ri._shallow_copy(values)

print(f"Result after _shallow_copy: {result}")
print(f"Result type: {type(result).__name__}")
print(f"Expected type: RangeIndex (for memory efficiency with equally-spaced values)")
print()

# Check if it's a RangeIndex
is_range_index = isinstance(result, RangeIndex)
print(f"Is result a RangeIndex? {is_range_index}")
print()

# Compare with multi-element case
print("=" * 50)
print("Comparison with 2-element RangeIndex:")
ri2 = RangeIndex(0, 2, 1)
values2 = np.array([0, 1])
result2 = ri2._shallow_copy(values2)
print(f"2-element values: {values2}")
print(f"2-element result type: {type(result2).__name__}")
print(f"Is 2-element result a RangeIndex? {isinstance(result2, RangeIndex)}")