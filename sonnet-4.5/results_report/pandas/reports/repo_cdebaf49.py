from pandas.io.json._normalize import convert_to_line_delimits

# Test case 1: Valid JSON array - should be processed
valid_array = '[1, 2, 3]'
result1 = convert_to_line_delimits(valid_array)
print(f"Input:  {repr(valid_array)}")
print(f"Output: {repr(result1)}")
print(f"Processed: {result1 != valid_array}")
print()

# Test case 2: Valid JSON object - should NOT be processed
valid_object = '{"a": 1}'
result2 = convert_to_line_delimits(valid_object)
print(f"Input:  {repr(valid_object)}")
print(f"Output: {repr(result2)}")
print(f"Unchanged: {result2 == valid_object}")
print()

# Test case 3: Malformed input (object start, array end) - should NOT be processed
malformed1 = '{"a": 1}]'
result3 = convert_to_line_delimits(malformed1)
print(f"Input:  {repr(malformed1)}")
print(f"Output: {repr(result3)}")
print(f"Unchanged: {result3 == malformed1}")
print()

# Test case 4: Malformed input (array start, object end) - should NOT be processed
malformed2 = '[1, 2, 3}'
result4 = convert_to_line_delimits(malformed2)
print(f"Input:  {repr(malformed2)}")
print(f"Output: {repr(result4)}")
print(f"Unchanged: {result4 == malformed2}")
print()

# Test case 5: Plain text - should NOT be processed
plain_text = 'plain text'
result5 = convert_to_line_delimits(plain_text)
print(f"Input:  {repr(plain_text)}")
print(f"Output: {repr(result5)}")
print(f"Unchanged: {result5 == plain_text}")