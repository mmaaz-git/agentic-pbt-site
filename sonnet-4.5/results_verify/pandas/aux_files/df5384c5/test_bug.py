import numpy as np
from pandas.arrays import SparseArray
from hypothesis import given, strategies as st

# Property-based test
@given(
    st.lists(st.integers(min_value=-10, max_value=10), min_size=1, max_size=20),
    st.integers(min_value=-10, max_value=10)
)
def test_nonzero_matches_dense(data, fill_value):
    arr = SparseArray(data, fill_value=fill_value)
    sparse_result = arr.nonzero()[0]
    dense_result = arr.to_dense().nonzero()[0]

    assert np.array_equal(sparse_result, dense_result), \
        f"sparse.nonzero() != to_dense().nonzero() for data={data}, fill_value={fill_value}"

# Test with the specific failing input
print("Testing with hypothesis...")
try:
    test_nonzero_matches_dense([0, 1, 2, 2], 2)
    print("Test passed")
except AssertionError as e:
    print(f"Test failed: {e}")

# Manual reproduction
print("\n" + "="*50)
print("Manual reproduction of the bug:")
print("="*50)

arr = SparseArray([0, 1, 2, 2], fill_value=2)

print(f"Array: {arr}")
print(f"Array data type: {type(arr)}")
print(f"Fill value: {arr.fill_value}")
print(f"to_dense(): {arr.to_dense()}")
print(f"to_dense() type: {type(arr.to_dense())}")
print(f"Expected nonzero positions (to_dense().nonzero()[0]): {arr.to_dense().nonzero()[0]}")
print(f"Actual nonzero positions (arr.nonzero()[0]): {arr.nonzero()[0]}")

try:
    assert np.array_equal(arr.nonzero()[0], arr.to_dense().nonzero()[0])
    print("\nAssertion passed - arrays are equal")
except AssertionError:
    print("\nAssertion failed - arrays are NOT equal")
    print(f"  Expected: {arr.to_dense().nonzero()[0]}")
    print(f"  Got:      {arr.nonzero()[0]}")