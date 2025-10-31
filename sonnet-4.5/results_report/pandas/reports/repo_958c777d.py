from pandas.api.types import is_re_compilable

# Test with single backslash - invalid regex pattern
print("Testing with single backslash '\\':")
result = is_re_compilable('\\')
print(f"Result: {result}")