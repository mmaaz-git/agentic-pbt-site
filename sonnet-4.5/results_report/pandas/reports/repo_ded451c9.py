import pandas.api.types as pat

# This should return False but instead crashes
result = pat.is_re_compilable(')')
print(f"Result: {result}")