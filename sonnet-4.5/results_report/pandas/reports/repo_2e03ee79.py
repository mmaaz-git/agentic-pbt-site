from pandas.compat._optional import get_version
import types

# Test 1: Non-string __version__ returns non-string value
mock_module = types.ModuleType("test_module")
mock_module.__version__ = 0

result = get_version(mock_module)
print(f"Test 1 - Non-string version:")
print(f"  Result: {result}")
print(f"  Type: {type(result)}")
print(f"  Expected type: str")
print(f"  Type matches expectation: {isinstance(result, str)}")
print()

# Test 2: psycopg2 module with non-string __version__ causes crash
psycopg2_module = types.ModuleType("psycopg2")
psycopg2_module.__version__ = 42

print("Test 2 - psycopg2 with non-string version:")
try:
    result = get_version(psycopg2_module)
    print(f"  Result: {result}")
    print(f"  Type: {type(result)}")
except AttributeError as e:
    print(f"  Error occurred: AttributeError: {e}")
    print(f"  This happens at line 81 where version.split() is called")