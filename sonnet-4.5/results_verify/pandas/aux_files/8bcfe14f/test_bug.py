"""Test the pandas indexes scoping bug"""
from pandas import Index
from pandas.core.indexes.api import union_indexes

# Test case 1: Basic union
idx1 = Index([1, 2, 3], dtype='int32')
idx2 = Index([4, 5, 6], dtype='int64')

result = union_indexes([idx1, idx2], sort=False)

print(f"idx1 dtype: {idx1.dtype}")
print(f"idx2 dtype: {idx2.dtype}")
print(f"Result dtype: {result.dtype}")
print(f"Result values: {result.values}")

# Test case 2: Let's also verify where _find_common_index_dtype is defined
import inspect

# Get the source of union_indexes
source = inspect.getsource(union_indexes)

# Check if the bug exists
if 'def _find_common_index_dtype(inds):' in source and '[idx.dtype for idx in indexes' in source:
    print("\nBUG CONFIRMED: Function parameter is 'inds' but code uses 'indexes' from outer scope")
else:
    print("\nBug not found in current version")