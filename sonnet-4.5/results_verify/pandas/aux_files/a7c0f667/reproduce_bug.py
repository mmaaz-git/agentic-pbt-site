from pandas.compat._optional import get_version
import types

mock_module = types.ModuleType("test_module")
mock_module.__version__ = 0

result = get_version(mock_module)
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Test what happens when psycopg2 has a non-string version
psycopg2_module = types.ModuleType("psycopg2")
psycopg2_module.__version__ = 123

try:
    psycopg2_result = get_version(psycopg2_module)
    print(f"psycopg2 result: {psycopg2_result}")
    print(f"psycopg2 type: {type(psycopg2_result)}")
except AttributeError as e:
    print(f"AttributeError occurred: {e}")