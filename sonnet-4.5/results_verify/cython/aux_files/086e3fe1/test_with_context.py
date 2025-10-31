from Cython.Build.Inline import safe_type, _create_context

class CustomClass:
    pass

# Test with default context
ctx = _create_context(('.',))
obj = CustomClass()

print("Testing safe_type with custom class instance and context...")
try:
    result = safe_type(obj, ctx)
    print(f"Result with context: {result}")
except Exception as e:
    print(f"Error with context: {type(e).__name__}: {e}")

# Test with None context (the bug)
print("\nTesting safe_type with custom class instance and None context...")
try:
    result = safe_type(obj, None)
    print(f"Result with None context: {result}")
except AttributeError as e:
    print(f"AttributeError with None context: {e}")