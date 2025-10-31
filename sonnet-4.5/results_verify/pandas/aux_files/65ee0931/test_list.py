from pandas.io.json._normalize import convert_to_line_delimits
import json

# Test with a JSON list (should be converted)
json_list = '[{"a": 1}, {"b": 2}]'
result = convert_to_line_delimits(json_list)
print(f"List Input:  {repr(json_list)}")
print(f"List Output: {repr(result)}")
print()

# Test with malformed JSON - starts with { but ends with ]
malformed = '{"a": 1]'
result_malformed = convert_to_line_delimits(malformed)
print(f"Malformed Input:  {repr(malformed)}")
print(f"Malformed Output: {repr(result_malformed)}")
print()

# Test the actual condition logic
def test_condition(s):
    original = not s[0] == "[" and s[-1] == "]"
    fixed = not (s[0] == "[" and s[-1] == "]")
    print(f"String: {repr(s)}")
    print(f"  Original condition (not s[0] == '[' and s[-1] == ']'): {original}")
    print(f"  Fixed condition (not (s[0] == '[' and s[-1] == ']')): {fixed}")
    print(f"  Should return early: {fixed}")
    print()

test_condition('{"key": "value"}')  # JSON object
test_condition('[{"a": 1}]')        # JSON list
test_condition('{"a": 1]')          # Malformed