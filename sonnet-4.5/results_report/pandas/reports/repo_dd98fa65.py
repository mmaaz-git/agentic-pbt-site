from pandas.io.json._normalize import convert_to_line_delimits

# Test 1: Empty string (causes crash)
print("Test 1: Empty string")
try:
    result = convert_to_line_delimits("")
    print(f"Result: '{result}'")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTest 2: String ending with ']' but not starting with '['")
result = convert_to_line_delimits("data]")
print(f"Input: 'data]'")
print(f"Result: '{result}'")
print(f"Returned unchanged: {result == 'data]'}")

print("\nTest 3: String starting with '[' but not ending with ']'")
result = convert_to_line_delimits("[incomplete")
print(f"Input: '[incomplete'")
print(f"Result: '{result}'")
print(f"Returned unchanged: {result == '[incomplete'}")

print("\nTest 4: Valid JSON array")
result = convert_to_line_delimits("[1,2,3]")
print(f"Input: '[1,2,3]'")
print(f"Result: '{result}'")