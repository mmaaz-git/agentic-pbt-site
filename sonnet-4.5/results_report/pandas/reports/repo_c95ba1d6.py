from pandas.io.json._normalize import convert_to_line_delimits

# Test with a simple JSON object
json_obj = '{"key": "value"}'
result = convert_to_line_delimits(json_obj)

print(f"Input:  {repr(json_obj)}")
print(f"Output: {repr(result)}")
print()

# Test with another JSON object
json_obj2 = '{"0": 0}'
result2 = convert_to_line_delimits(json_obj2)

print(f"Input:  {repr(json_obj2)}")
print(f"Output: {repr(result2)}")
print()

# Show the issue: the function should return non-lists unchanged
assert json_obj == result, f"Expected unchanged output for JSON object, but got corrupted JSON"