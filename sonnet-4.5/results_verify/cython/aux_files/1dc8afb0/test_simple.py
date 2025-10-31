#!/usr/bin/env python3
from Cython.Build.Dependencies import parse_list

print("Testing parse_list with empty string in double quotes")
try:
    result = parse_list('[""]')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting parse_list with empty string in single quotes")
try:
    result = parse_list("['']")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting parse_list with non-empty string")
try:
    result = parse_list('["hello"]')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\nTesting parse_list with multiple items including empty string")
try:
    result = parse_list('["a", "", "b"]')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")