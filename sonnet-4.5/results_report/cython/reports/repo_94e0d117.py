from Cython.Utils import strip_py2_long_suffix

# Test the function with an empty string
try:
    result = strip_py2_long_suffix('')
    print(f"Result: {result!r}")
except Exception as e:
    print(f"{e.__class__.__name__}: {e}")