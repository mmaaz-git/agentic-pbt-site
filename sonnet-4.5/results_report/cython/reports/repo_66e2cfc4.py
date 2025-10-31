from Cython.Build.Dependencies import parse_list

# Test case with unclosed single quote
try:
    result = parse_list("'")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

# Test case with unclosed double quote
print("\nTest with unclosed double quote:")
try:
    result = parse_list('"')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

# Test case with unclosed quote and text
print("\nTest with unclosed quote and text:")
try:
    result = parse_list("'hello")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")