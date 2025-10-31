from Cython.Build.Inline import safe_type

class CustomClass:
    pass

obj = CustomClass()
result = safe_type(obj)
print(f"Result: {result}")