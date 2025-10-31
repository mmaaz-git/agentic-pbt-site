import numpy as np
import traceback

print("Testing bug reproduction...")
print("=" * 50)

# Test 1: Basic reproduction as described in bug report
print("\nTest 1: Basic reproduction")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', gdict={'A': A, 'B': B})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    traceback.print_exc()

print("\n" + "=" * 50)

# Test 2: Verify it works when both ldict and gdict are provided
print("\nTest 2: With both ldict and gdict")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B', ldict={'A': A, 'B': B}, gdict={})
    print(f"Result:\n{result}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)

# Test 3: Verify normal usage (without explicit dicts) works
print("\nTest 3: Normal usage (implicit scope)")
try:
    A = np.matrix([[1, 2], [3, 4]])
    B = np.matrix([[5, 6], [7, 8]])
    result = np.bmat('A,B')
    print(f"Result:\n{result}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)

# Test 4: Run the hypothesis test
print("\nTest 4: Property-based test")
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
    try:
        result = np.bmat('X,Y', gdict=global_vars)
        expected = np.bmat([[m1, m2]])
        assert np.array_equal(result, expected)
        return True
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            print(f"Confirmed bug: {e}")
            return False
        raise

# Run a few samples
import random
random.seed(42)
for i in range(3):
    arr1 = np.random.randn(2, 2)
    arr2 = np.random.randn(2, 2)
    print(f"\nSample {i+1}:")
    test_bmat_gdict_without_ldict_crashes(arr1, arr2)