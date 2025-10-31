from pandas.io.json._normalize import convert_to_line_delimits

json_obj = '{"key": "value"}'
result = convert_to_line_delimits(json_obj)

print(f"Input:  {repr(json_obj)}")
print(f"Output: {repr(result)}")

assert json_obj == result, f"Expected unchanged output for non-list JSON"