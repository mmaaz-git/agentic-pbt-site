from pandas.compat._optional import import_optional_dependency

# Test 1: Module with version too old
print("Test 1: hypothesis with errors='ignore' and impossibly high min_version")
result = import_optional_dependency("hypothesis", errors="ignore", min_version="999.0.0")
print(f"Result: {result}")
print(f"Result is None: {result is None}")
print()

# Test 2: Module without __version__ attribute
print("Test 2: sys with errors='ignore' and min_version")
try:
    result2 = import_optional_dependency("sys", errors="ignore", min_version="1.0.0")
    print(f"Result2: {result2}")
    print(f"Result2 is None: {result2 is None}")
except ImportError as e:
    print(f"Raised ImportError: {e}")