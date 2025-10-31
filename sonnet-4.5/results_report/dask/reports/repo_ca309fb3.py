import dask.utils

# Test with empty string
try:
    result = dask.utils.parse_timedelta('')
    print(f"Empty string result: {result}")
except Exception as e:
    print(f"Empty string error: {type(e).__name__}: {e}")

# Test with space-only string
try:
    result = dask.utils.parse_timedelta(' ')
    print(f"Space-only string result: {result}")
except Exception as e:
    print(f"Space-only string error: {type(e).__name__}: {e}")

# Test with multiple spaces
try:
    result = dask.utils.parse_timedelta('   ')
    print(f"Multiple spaces result: {result}")
except Exception as e:
    print(f"Multiple spaces error: {type(e).__name__}: {e}")