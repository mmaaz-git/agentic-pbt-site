#!/usr/bin/env python3
"""Reproduce the parse_list bug with unclosed quotes."""

from Cython.Build.Dependencies import parse_list

# Test case 1: single double-quote
print("Test 1: Single double-quote")
try:
    result = parse_list('"')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Other error ({type(e).__name__}): {e}")

# Test case 2: single quote
print("\nTest 2: Single quote")
try:
    result = parse_list("'")
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Other error ({type(e).__name__}): {e}")

# Test case 3: double quotes without content
print("\nTest 3: Two double-quotes")
try:
    result = parse_list('""')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Other error ({type(e).__name__}): {e}")

# Test case 4: two single quotes
print("\nTest 4: Two single-quotes")
try:
    result = parse_list("''")
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Other error ({type(e).__name__}): {e}")