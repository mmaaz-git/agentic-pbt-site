import pandas.api.types as pat

# Test invalid regex pattern that should return False
# but instead raises re.PatternError
result = pat.is_re_compilable(')')
print(f"Result: {result}")