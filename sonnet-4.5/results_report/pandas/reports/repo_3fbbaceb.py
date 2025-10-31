import pandas.api.types as pat

# Test with invalid regex pattern
result = pat.is_re_compilable('[')
print(f"Result: {result}")