import numpy as np
from hypothesis import given, strategies as st, settings

# Test 1: Strided arrays test
@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=2, max_size=100))
@settings(max_examples=10)  # Reduced for testing
def test_as_ctypes_accepts_strided_arrays(data):
    np_array = np.array(data, dtype=np.int32)
    sliced = np_array[::2]

    try:
        ctypes_array = np.ctypeslib.as_ctypes(sliced)
        print(f"SUCCESS: Strided array was accepted for data {data[:5]}...")
        assert True, "Strided array was accepted"
    except TypeError as e:
        print(f"FAILURE: Strided array rejected: {e}")
        assert False, f"as_ctypes docstring says 'anything that exposes __array_interface__ is accepted', but strided arrays are rejected: {e}"

# Test 2: Readonly arrays test
@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
@settings(max_examples=10)  # Reduced for testing
def test_as_ctypes_accepts_readonly_arrays(data):
    np_array = np.array(data, dtype=np.int32)
    np_array.flags.writeable = False

    try:
        ctypes_array = np.ctypeslib.as_ctypes(np_array)
        print(f"SUCCESS: Readonly array was accepted for data {data[:5]}...")
        assert True, "Readonly array was accepted"
    except TypeError as e:
        print(f"FAILURE: Readonly array rejected: {e}")
        assert False, f"as_ctypes docstring says 'anything that exposes __array_interface__ is accepted', but readonly arrays are rejected: {e}"

# Manual reproduction tests
print("=" * 60)
print("Manual Test 1: Strided array (sliced)")
print("=" * 60)
try:
    arr = np.array([1, 2, 3, 4, 5, 6])
    sliced = arr[::2]
    print(f"Original array: {arr}")
    print(f"Sliced array (every 2nd element): {sliced}")
    print(f"Sliced array has __array_interface__: {hasattr(sliced, '__array_interface__')}")

    result = np.ctypeslib.as_ctypes(sliced)
    print(f"SUCCESS: as_ctypes worked for sliced array")
except TypeError as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("Manual Test 2: Readonly array")
print("=" * 60)
try:
    arr = np.array([1, 2, 3])
    arr.flags.writeable = False
    print(f"Array: {arr}")
    print(f"Array is readonly: {not arr.flags.writeable}")
    print(f"Array has __array_interface__: {hasattr(arr, '__array_interface__')}")

    result = np.ctypeslib.as_ctypes(arr)
    print(f"SUCCESS: as_ctypes worked for readonly array")
except TypeError as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("Manual Test 3: Regular contiguous array (should work)")
print("=" * 60)
try:
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Array: {arr}")
    print(f"Array has __array_interface__: {hasattr(arr, '__array_interface__')}")

    result = np.ctypeslib.as_ctypes(arr)
    print(f"SUCCESS: as_ctypes worked for regular array")
    print(f"Result type: {type(result)}")
    print(f"Result values: {result[:]}")
except TypeError as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("Running Hypothesis tests...")
print("=" * 60)
print("\nTest 1: Strided arrays")
try:
    test_as_ctypes_accepts_strided_arrays()
    print("All strided array tests passed!")
except AssertionError as e:
    print(f"Strided array test failed as expected")

print("\nTest 2: Readonly arrays")
try:
    test_as_ctypes_accepts_readonly_arrays()
    print("All readonly array tests passed!")
except AssertionError as e:
    print(f"Readonly array test failed as expected")