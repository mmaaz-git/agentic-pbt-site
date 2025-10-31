from pandas.api.types import is_re_compilable

# Test invalid regex pattern that should crash
pattern = '?'
result = is_re_compilable(pattern)
print(f"Result for pattern '{pattern}': {result}")