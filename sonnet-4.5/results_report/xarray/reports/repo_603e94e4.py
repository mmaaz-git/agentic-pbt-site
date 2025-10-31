from xarray.core.utils import is_uniform_spaced

# Test empty array
try:
    result = is_uniform_spaced([])
    print(f"Empty array result: {result}")
except Exception as e:
    print(f"Empty array error: {type(e).__name__}: {e}")

# Test single element array
try:
    result = is_uniform_spaced([5])
    print(f"Single element array result: {result}")
except Exception as e:
    print(f"Single element array error: {type(e).__name__}: {e}")

# Test two element array (should work)
try:
    result = is_uniform_spaced([1, 2])
    print(f"Two element array result: {result}")
except Exception as e:
    print(f"Two element array error: {type(e).__name__}: {e}")

# Test normal array (should work)
try:
    result = is_uniform_spaced([1, 2, 3, 4, 5])
    print(f"Normal array result: {result}")
except Exception as e:
    print(f"Normal array error: {type(e).__name__}: {e}")