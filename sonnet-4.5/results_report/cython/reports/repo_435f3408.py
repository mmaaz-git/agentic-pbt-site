from Cython.Build.Dependencies import parse_list

# Test with single double quote
try:
    result = parse_list('"')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError with double quote: {e}")

# Test with single single quote
try:
    result = parse_list("'")
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError with single quote: {e}")