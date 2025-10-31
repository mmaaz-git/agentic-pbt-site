from pandas import RangeIndex, Index
from hypothesis import given, strategies as st, assume

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

# Test the hypothesis test case
print("\n=== Hypothesis Test Case ===")
@given(
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=1, max_value=20),
    st.text(min_size=1, max_size=20),
    st.text(min_size=1, max_size=20),
)
def test_concat_single_index_renames(start, stop, step, original_name, new_name):
    assume(start != stop)
    assume(original_name != new_name)

    idx = RangeIndex(start, stop, step, name=original_name)
    assume(len(idx) > 0)

    result = idx._concat([idx], name=new_name)

    assert result.name == new_name, f"_concat should rename to '{new_name}' but got '{result.name}'"

# Test the specific failing input
print("Testing specific failing input: start=0, stop=1, step=1, original_name='0', new_name='6'")
try:
    test_concat_single_index_renames(0, 1, 1, '0', '6')
    print("Test PASSED")
except AssertionError as e:
    print(f"Test FAILED: {e}")

# Run the full hypothesis test
print("\n=== Running Hypothesis Tests ===")
try:
    test_concat_single_index_renames()
    print("All hypothesis tests PASSED")
except Exception as e:
    print(f"Hypothesis tests FAILED: {e}")

# Test other edge cases
print("\n=== Testing Other Cases ===")

# Test with multiple indexes
print("\n1. Multiple indexes (should apply name):")
idx1 = RangeIndex(0, 3, name="idx1")
idx2 = RangeIndex(3, 6, name="idx2")
result_multi = idx1._concat([idx1, idx2], name="combined")
print(f"   Result name: '{result_multi.name}' (expected: 'combined')")

# Test with empty index
print("\n2. Empty index (should apply name):")
empty_idx = RangeIndex(0, 0, name="empty")
result_empty = empty_idx._concat([empty_idx], name="renamed")
print(f"   Result name: '{result_empty.name}' (expected: 'renamed')")

# Test with None name
print("\n3. Single index with None name:")
idx_none = RangeIndex(0, 3, name="original")
result_none = idx_none._concat([idx_none], name=None)
print(f"   Result name: {result_none.name} (expected: None)")