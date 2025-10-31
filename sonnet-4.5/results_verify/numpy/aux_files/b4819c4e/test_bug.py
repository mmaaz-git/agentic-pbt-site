import numpy as np
import sys
import traceback

print("Testing numpy.bmat with gdict but no ldict")
print("=" * 60)

# Test 1: Simple reproduction case
print("\nTest 1: Simple reproduction case")
print("Code: np.bmat('A,B', gdict={'A': A, 'B': B})")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', gdict={'A': A, 'B': B})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: With both ldict and gdict (should work)
print("\nTest 2: With both ldict and gdict (should work)")
print("Code: np.bmat('A,B', ldict={'A': A, 'B': B}, gdict={})")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', ldict={'A': A, 'B': B}, gdict={})
    print(f"Result:\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 3: With ldict as empty dict and gdict provided
print("\nTest 3: With ldict as empty dict and gdict with values")
print("Code: np.bmat('A,B', ldict={}, gdict={'A': A, 'B': B})")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', ldict={}, gdict={'A': A, 'B': B})
    print(f"Result:\n{result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 4: Property-based test from bug report
print("\nTest 4: Running the property-based test")
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
    npst.arrays(dtype=np.float64, shape=(2, 2), elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
)
def test_bmat_gdict_without_ldict_crashes(arr1, arr2):
    m1 = np.matrix(arr1)
    m2 = np.matrix(arr2)
    global_vars = {'X': m1, 'Y': m2}
    result = np.bmat('X,Y', gdict=global_vars)
    expected = np.bmat([[m1, m2]])
    assert np.array_equal(result, expected)

try:
    test_bmat_gdict_without_ldict_crashes()
    print("Property-based test passed!")
except Exception as e:
    print(f"Property-based test failed: {type(e).__name__}: {e}")

# Test 5: Expected behavior comparison
print("\nTest 5: Expected behavior comparison")
print("Comparing np.bmat([[A, B]]) with np.bmat('A,B', gdict={'A': A, 'B': B})")
A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[5, 6], [7, 8]])
expected = np.bmat([[A, B]])
print(f"Expected result using list syntax:\n{expected}")