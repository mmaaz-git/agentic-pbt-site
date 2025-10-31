import numpy as np
from xarray.compat.array_api_compat import result_type, is_weak_scalar_type

# Test 1: Verify that string is recognized as a weak scalar type
print("Test 1: is_weak_scalar_type('test'):", is_weak_scalar_type("test"))

# Test 2: Try to call result_type with a string scalar
print("\nTest 2: Calling result_type('test', xp=np)")
try:
    result = result_type("test", xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test 3: Try to call result_type with a bytes scalar
print("\nTest 3: Calling result_type(b'test', xp=np)")
try:
    result = result_type(b"test", xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test 4: Verify that numpy's result_type doesn't support strings
print("\nTest 4: Calling np.result_type('test') directly")
try:
    result = np.result_type("test")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

# Test 5: Verify that _future_array_api_result_type works correctly
print("\nTest 5: Calling _future_array_api_result_type('test', xp=np)")
try:
    from xarray.compat.array_api_compat import _future_array_api_result_type
    result = _future_array_api_result_type("test", xp=np)
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")