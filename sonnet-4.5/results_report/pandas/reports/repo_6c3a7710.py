from pandas.io.json._normalize import convert_to_line_delimits

# Test with single digit
print(f"Input: '0' -> Output: {repr(convert_to_line_delimits('0'))}")

# Test with multi-digit number
print(f"Input: '123' -> Output: {repr(convert_to_line_delimits('123'))}")

# Test with empty JSON object
print(f"Input: '{{}}' -> Output: {repr(convert_to_line_delimits('{}'))}")

# Test with JSON string
print(f"Input: '\"hello\"' -> Output: {repr(convert_to_line_delimits('\"hello\"'))}")

# Test with actual JSON array (should be processed)
print(f"Input: '[1,2,3]' -> Output: {repr(convert_to_line_delimits('[1,2,3]'))}")

# Test with JSON object
print(f"Input: '{{\"a\":1}}' -> Output: {repr(convert_to_line_delimits('{\"a\":1}'))}")