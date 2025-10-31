from pandas import Index
from pandas.core.indexes.api import union_indexes

print("=== Comparing Index.union() vs union_indexes() behavior ===\n")

# Test 1: Index with duplicates [0, 0]
idx1 = Index([0, 0])
print(f"Test 1: idx = Index([0, 0])")
print(f"idx.union(idx) = {list(idx1.union(idx1))}")
print(f"union_indexes([idx, idx]) = {list(union_indexes([idx1, idx1]))}")
print(f"Same behavior? {list(idx1.union(idx1)) == list(union_indexes([idx1, idx1]))}\n")

# Test 2: Different multiplicities
idx2 = Index([1, 1, 1, 2])
idx3 = Index([1, 1, 2, 2, 3])
print(f"Test 2: Multiset union behavior")
print(f"idx1 = Index([1, 1, 1, 2])")
print(f"idx2 = Index([1, 1, 2, 2, 3])")
print(f"idx1.union(idx2) = {list(idx2.union(idx3))}")
print(f"union_indexes([idx1, idx2]) = {list(union_indexes([idx2, idx3]))}")
print(f"Expected multiset: [1, 1, 1, 2, 2, 3] (max multiplicity)")

# Test 3: Check if union_indexes with different indexes removes duplicates
idx4 = Index([1, 1, 2])
idx5 = Index([2, 3, 3])
print(f"\nTest 3: Different indexes")
print(f"idx1 = Index([1, 1, 2])")
print(f"idx2 = Index([2, 3, 3])")
print(f"idx1.union(idx2) = {list(idx4.union(idx5))}")
print(f"union_indexes([idx1, idx2]) = {list(union_indexes([idx4, idx5]))}")

# Test 4: Check what _unique_indices would do
print(f"\nTest 4: What _unique_indices is supposed to do")
print("Based on the code, _unique_indices calls .unique() on the first index")
print("and then adds unique elements from other indexes.")
idx6 = Index([1, 1, 2, 2])
print(f"Index([1, 1, 2, 2]).unique() = {list(idx6.unique())}")