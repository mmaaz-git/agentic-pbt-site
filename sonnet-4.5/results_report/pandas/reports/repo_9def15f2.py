from pandas.io.excel._base import inspect_excel_format

# Test with empty bytes
print("Testing inspect_excel_format with empty bytes b''")
print("According to docstring, this should raise ValueError('stream is empty')")
print()

try:
    result = inspect_excel_format(b'')
    print(f"Result: {result}")
    print(f"Type of result: {type(result)}")
    print("ERROR: No exception was raised! Expected ValueError.")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")
except Exception as e:
    print(f"Unexpected exception: {type(e).__name__}: {e}")