import pandas.arrays as pa
import numpy as np
import sys

# Test 1: Simple case from bug report
print("Test 1: SparseArray([0], fill_value=0)")
try:
    sparse = pa.SparseArray([0], fill_value=0)
    print(f"Created sparse array: {sparse}")
    print(f"Fill value: {sparse.fill_value}")
    print(f"_null_fill_value: {sparse._null_fill_value}")

    # Set recursion limit to detect infinite recursion quickly
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100)

    result = sparse.cumsum()
    print(f"Result: {result}")
    print(f"Result fill_value: {result.fill_value}")

    sys.setrecursionlimit(old_limit)
except RecursionError as e:
    print(f"RecursionError: {e}")
    sys.setrecursionlimit(1000)
except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 2: Test with multiple values
print("Test 2: SparseArray([0, 1, 0, 2], fill_value=0)")
try:
    sparse = pa.SparseArray([0, 1, 0, 2], fill_value=0)
    print(f"Created sparse array: {sparse}")
    print(f"Fill value: {sparse.fill_value}")
    print(f"_null_fill_value: {sparse._null_fill_value}")

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100)

    result = sparse.cumsum()
    print(f"Result: {result}")
    print(f"Result fill_value: {result.fill_value}")

    sys.setrecursionlimit(old_limit)
except RecursionError as e:
    print(f"RecursionError: {e}")
    sys.setrecursionlimit(1000)
except Exception as e:
    print(f"Other error: {e}")

print("\n" + "="*50 + "\n")

# Test 3: Test with NaN fill value (should work according to code)
print("Test 3: SparseArray([0, 1, np.nan, 2], fill_value=np.nan)")
try:
    sparse = pa.SparseArray([0, 1, np.nan, 2], fill_value=np.nan)
    print(f"Created sparse array: {sparse}")
    print(f"Fill value: {sparse.fill_value}")
    print(f"_null_fill_value: {sparse._null_fill_value}")

    result = sparse.cumsum()
    print(f"Result: {result}")
    print(f"Result fill_value: {result.fill_value}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50 + "\n")

# Test 4: Test that verifies the documentation claim about fill_value
print("Test 4: Verify documentation claim about fill_value")
try:
    # According to docs: "the fill value will be `np.nan` regardless"
    sparse1 = pa.SparseArray([1, 2, 3, 0, 4], fill_value=0)
    sparse2 = pa.SparseArray([1, 2, 3, np.nan, 4], fill_value=np.nan)

    print(f"Sparse1 (fill_value=0): {sparse1}")
    print(f"Sparse2 (fill_value=nan): {sparse2}")

    # This should fail for sparse1 due to recursion issue
    try:
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)
        result1 = sparse1.cumsum()
        sys.setrecursionlimit(old_limit)
        print(f"Result1 fill_value: {result1.fill_value}")
    except RecursionError:
        print("Sparse1 causes RecursionError as expected")
        sys.setrecursionlimit(1000)

    # This should work for sparse2
    result2 = sparse2.cumsum()
    print(f"Result2: {result2}")
    print(f"Result2 fill_value: {result2.fill_value}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 5: Property-based test from the bug report
print("Test 5: Property-based test with hypothesis")
try:
    from hypothesis import given, strategies as st, settings

    def sparse_array_strategy(min_size=0, max_size=50):
        @st.composite
        def _strat(draw):
            size = draw(st.integers(min_value=min_size, max_value=max_size))
            fill_value = draw(st.sampled_from([0, 0.0, -1, 1]))
            values = draw(st.lists(
                st.sampled_from([fill_value, 1, 2, 3, -1, 10]),
                min_size=size, max_size=size
            ))
            return pa.SparseArray(values, fill_value=fill_value)
        return _strat()

    @given(sparse_array_strategy(min_size=1, max_size=20))
    @settings(max_examples=10, deadline=None)
    def test_sparsearray_cumsum_doesnt_crash(arr):
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)
        try:
            result = arr.cumsum()
            assert isinstance(result, pa.SparseArray)
            sys.setrecursionlimit(old_limit)
        except RecursionError:
            sys.setrecursionlimit(old_limit)
            raise

    test_sparsearray_cumsum_doesnt_crash()
    print("Property test passed (no crashes)")

except Exception as e:
    print(f"Property test failed: {e}")
    import traceback
    traceback.print_exc()