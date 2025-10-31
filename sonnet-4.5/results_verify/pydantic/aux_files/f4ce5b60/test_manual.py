from flask.helpers import get_root_path
import os

print("Testing get_root_path with 'sys' module...")
try:
    result = get_root_path('sys')
    print(f"Result: {result}")
    print(f"Current working directory: {os.getcwd()}")
except RuntimeError as e:
    print(f"RuntimeError raised: {e}")

print("\nTesting get_root_path with nonexistent module...")
try:
    result = get_root_path('this_module_does_not_exist_1234567890')
    print(f"Result: {result}")
    print(f"Current working directory: {os.getcwd()}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")