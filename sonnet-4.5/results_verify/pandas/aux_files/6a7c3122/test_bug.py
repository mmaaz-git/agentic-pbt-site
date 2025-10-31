from hypothesis import given, strategies as st, settings
from pandas.io.json._normalize import convert_to_line_delimits

# Test 1: Property-based test from the bug report
@given(st.text(min_size=2, max_size=100))
@settings(max_examples=1000)
def test_convert_to_line_delimits_only_processes_arrays(s):
    result = convert_to_line_delimits(s)

    if s[0] == '[' and s[-1] == ']':
        pass
    else:
        assert result == s, f"Non-array input should be returned unchanged"

# Test 2: Specific failing case from the bug report
malformed_input = '{"a": 1}]'
result = convert_to_line_delimits(malformed_input)

print("Test case from bug report:")
print(f"Input:  {repr(malformed_input)}")
print(f"Output: {repr(result)}")
print(f"Unchanged: {result == malformed_input}")
print()

# Additional test cases to understand the behavior
test_cases = [
    '[1, 2, 3]',     # Valid JSON array
    '{"a": 1}',      # Valid JSON object
    '{"a": 1}]',     # Malformed: object start, array end
    '[1, 2, 3}',     # Malformed: array start, object end
    '{[1, 2]}',      # Malformed: object with array inside
    'plain text',    # Not JSON at all
]

print("Additional test cases:")
for test_input in test_cases:
    try:
        output = convert_to_line_delimits(test_input)
        print(f"Input: {repr(test_input):<20} -> Output: {repr(output):<20} (unchanged: {output == test_input})")
    except Exception as e:
        print(f"Input: {repr(test_input):<20} -> Error: {e}")

# Now run the hypothesis test
print("\nRunning property-based test...")
try:
    test_convert_to_line_delimits_only_processes_arrays()
except AssertionError as e:
    print(f"Property test failed with: {e}")