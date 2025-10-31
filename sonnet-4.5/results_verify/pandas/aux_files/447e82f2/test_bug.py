from hypothesis import given, strategies as st, settings
from pandas.io.json._normalize import convert_to_line_delimits

print("Testing the property-based test:")
print("-" * 50)

@settings(max_examples=500)
@given(st.text(min_size=2))
def test_convert_to_line_delimits_only_processes_json_arrays(s):
    result = convert_to_line_delimits(s)
    is_json_array_format = s[0] == '[' and s[-1] == ']'

    if not is_json_array_format:
        assert result == s, (
            f"Non-JSON-array string should be returned unchanged. "
            f"Input: {s!r}, Output: {result!r}"
        )

# Run the hypothesis test
try:
    test_convert_to_line_delimits_only_processes_json_arrays()
    print("Property-based test PASSED")
except AssertionError as e:
    print(f"Property-based test FAILED: {e}")

print("\n" + "=" * 50)
print("Testing specific examples:")
print("-" * 50)

# Test the specific examples
result1 = convert_to_line_delimits("abc")
print(f"Input: 'abc', Output: {result1!r}")
print(f"Expected: 'abc', Got: {result1!r}, Match: {result1 == 'abc'}")

result2 = convert_to_line_delimits("[invalid")
print(f"\nInput: '[invalid', Output: {result2!r}")
print(f"Expected: '[invalid', Got: {result2!r}, Match: {result2 == '[invalid'}")

result3 = convert_to_line_delimits("test]")
print(f"\nInput: 'test]', Output: {result3!r}")
print(f"Expected: 'test]', Got: {result3!r}, Match: {result3 == 'test]'}")

result4 = convert_to_line_delimits("[valid]")
print(f"\nInput: '[valid]', Output: {result4!r}")
print(f"This should be processed as a JSON array")

print("\n" + "=" * 50)
print("Testing operator precedence analysis:")
print("-" * 50)

# Let's verify the operator precedence claim
s = "abc"
print(f"\nFor s = 'abc':")
print(f"s[0] = {s[0]!r}, s[-1] = {s[-1]!r}")
print(f"s[0] == '[' = {s[0] == '['}")
print(f"s[-1] == ']' = {s[-1] == ']'}")
print(f"not s[0] == '[' = {not s[0] == '['}")
print(f"(not s[0] == '[') and (s[-1] == ']') = {(not s[0] == '[') and (s[-1] == ']')}")

s = "[invalid"
print(f"\nFor s = '[invalid':")
print(f"s[0] = {s[0]!r}, s[-1] = {s[-1]!r}")
print(f"s[0] == '[' = {s[0] == '['}")
print(f"s[-1] == ']' = {s[-1] == ']'}")
print(f"not s[0] == '[' = {not s[0] == '['}")
print(f"(not s[0] == '[') and (s[-1] == ']') = {(not s[0] == '[') and (s[-1] == ']')}")

s = "test]"
print(f"\nFor s = 'test]':")
print(f"s[0] = {s[0]!r}, s[-1] = {s[-1]!r}")
print(f"s[0] == '[' = {s[0] == '['}")
print(f"s[-1] == ']' = {s[-1] == ']'}")
print(f"not s[0] == '[' = {not s[0] == '['}")
print(f"(not s[0] == '[') and (s[-1] == ']') = {(not s[0] == '[') and (s[-1] == ']')}")