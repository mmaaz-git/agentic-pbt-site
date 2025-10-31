from flask.helpers import get_root_path

# Test with built-in module 'sys'
try:
    result = get_root_path('sys')
    print(f"Result for 'sys': {result}")
except Exception as e:
    print(f"Error for 'sys': {type(e).__name__}: {e}")