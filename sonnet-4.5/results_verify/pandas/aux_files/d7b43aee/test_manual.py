from pandas import Index
from pandas.core.indexes.api import union_indexes

print("=== Manual Reproduction Test ===\n")

# Test 1: Index with duplicates [0, 0]
idx_with_dups = Index([0, 0])
result = union_indexes([idx_with_dups, idx_with_dups])
print(f"Test 1: union_indexes([Index([0, 0]), Index([0, 0])])")
print(f"Result: {list(result)}")
print(f"Expected: [0]")
print(f"Actual contains duplicates: {len(result) > 1}")
print(f"Bug reproduced: {list(result) == [0, 0]}\n")

# Test 2: Index with duplicates [1, 1]
idx1 = Index([1, 1])
idx2 = Index([1, 1])
result2 = union_indexes([idx1, idx2])
print(f"Test 2: union_indexes([Index([1, 1]), Index([1, 1])])")
print(f"Result: {list(result2)}")
print(f"Expected: [1]")
print(f"Still has duplicates: {len(result2) > 1}")
print(f"Bug reproduced: {list(result2) == [1, 1]}\n")

# Test 3: Compare with standard Index.union() method
idx3 = Index([2, 2, 3])
print(f"Test 3: Compare with Index.union() method")
print(f"Index([2, 2, 3]).union(Index([2, 2, 3])): {list(idx3.union(idx3))}")
print(f"union_indexes([Index([2, 2, 3]), Index([2, 2, 3])]): {list(union_indexes([idx3, idx3]))}")
print(f"Results differ: {list(idx3.union(idx3)) != list(union_indexes([idx3, idx3]))}\n")

# Test 4: Test when indexes are different (should work correctly)
idx4 = Index([1, 1, 2])
idx5 = Index([2, 2, 3])
result4 = union_indexes([idx4, idx5])
print(f"Test 4: Different indexes - union_indexes([Index([1, 1, 2]), Index([2, 2, 3])])")
print(f"Result: {list(result4)}")
print(f"Expected (unique values): [1, 2, 3]")
print(f"Correctly removes duplicates: {sorted(list(result4)) == [1, 2, 3]}")