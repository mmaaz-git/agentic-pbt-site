from pandas import RangeIndex, Index

# First, let's test the basic reproduction
print("=== Basic Reproduction Test ===")
idx = RangeIndex(0, 3, name="original")
result = idx._concat([idx], name="new_name")

print(f"Expected name: 'new_name'")
print(f"Actual name: '{result.name}'")
print(f"Names match: {result.name == 'new_name'}")

base_idx = Index([0, 1, 2], name="original")
base_result = base_idx._concat([base_idx], name="new_name")
print(f"\nBase Index class name: '{base_result.name}'")
print(f"Base Index names match: {base_result.name == 'new_name'}")

# Test the specific failing input
print("\n=== Testing specific failing input ===")
print("start=0, stop=1, step=1, original_name='0', new_name='6'")
idx_test = RangeIndex(0, 1, 1, name='0')
result_test = idx_test._concat([idx_test], name='6')
print(f"Expected name: '6'")
print(f"Actual name: '{result_test.name}'")
print(f"Names match: {result_test.name == '6'}")

# Test other edge cases
print("\n=== Testing Other Cases ===")

# Test with multiple indexes
print("\n1. Multiple consecutive indexes (should apply name):")
idx1 = RangeIndex(0, 3, name="idx1")
idx2 = RangeIndex(3, 6, name="idx2")
result_multi = idx1._concat([idx1, idx2], name="combined")
print(f"   Result name: '{result_multi.name}' (expected: 'combined')")
print(f"   Names match: {result_multi.name == 'combined'}")

# Test with non-consecutive indexes
print("\n2. Non-consecutive indexes (should apply name):")
idx1 = RangeIndex(0, 3, name="idx1")
idx2 = RangeIndex(5, 8, name="idx2")
result_noncons = idx1._concat([idx1, idx2], name="noncons")
print(f"   Result name: '{result_noncons.name}' (expected: 'noncons')")
print(f"   Names match: {result_noncons.name == 'noncons'}")

# Test with empty index
print("\n3. Empty index (should apply name):")
empty_idx = RangeIndex(0, 0, name="empty")
result_empty = empty_idx._concat([empty_idx], name="renamed")
print(f"   Result name: '{result_empty.name}' (expected: 'renamed')")
print(f"   Names match: {result_empty.name == 'renamed'}")

# Test with None name
print("\n4. Single index with None name:")
idx_none = RangeIndex(0, 3, name="original")
result_none = idx_none._concat([idx_none], name=None)
print(f"   Result name: {result_none.name} (expected: None)")
print(f"   Names match: {result_none.name is None}")

# Test behavior consistency
print("\n=== Behavior Consistency Test ===")
print("Testing if behavior is consistent across different input sizes:")

# Single index
idx = RangeIndex(0, 5, name="test")
result1 = idx._concat([idx], name="new")
print(f"Single index: name='{result1.name}' (expected: 'new')")

# Same index twice
result2 = idx._concat([idx, idx], name="new")
print(f"Two indexes: name='{result2.name}' (expected: 'new')")

# Three of the same index
result3 = idx._concat([idx, idx, idx], name="new")
print(f"Three indexes: name='{result3.name}' (expected: 'new')")

print(f"\nConsistency check: All should have name='new'")
print(f"  Single: {result1.name == 'new'}")
print(f"  Two: {result2.name == 'new'}")
print(f"  Three: {result3.name == 'new'}")