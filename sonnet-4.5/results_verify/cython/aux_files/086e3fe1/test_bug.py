from Cython.Build.Inline import safe_type

class CustomClass:
    pass

obj = CustomClass()
print("Testing safe_type with custom class instance...")
try:
    result = safe_type(obj)
    print(f"Result: {result}")
except AttributeError as e:
    print(f"AttributeError occurred: {e}")
except Exception as e:
    print(f"Other exception occurred: {type(e).__name__}: {e}")