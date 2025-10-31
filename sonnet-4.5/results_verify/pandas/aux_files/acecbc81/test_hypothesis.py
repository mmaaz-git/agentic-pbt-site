#!/usr/bin/env python3
"""Run the exact Hypothesis test from the bug report"""

import numpy as np
from pandas.core.util.hashing import hash_array
import traceback

print("Testing the exact code from the bug report:")
print("=" * 60)

# First test the direct reproduction
print("\n1. Direct reproduction code:")
print("-" * 40)
print("Code: arr = np.array(['\\ud800'], dtype=object)")
print("      hash_array(arr)")
print()

try:
    arr = np.array(['\ud800'], dtype=object)
    result = hash_array(arr)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Failed with {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Now test what UTF-8 actually says about surrogates
print("\n" + "=" * 60)
print("Understanding surrogate characters:")
print("-" * 40)

print("\nSurrogate characters (U+D800 to U+DFFF) are:")
print("- Valid in Python strings")
print("- Used internally for UTF-16 encoding")
print("- INVALID in UTF-8 encoding")
print()

print("Python's behavior with surrogates:")
s = '\ud800'
print(f"  String: {repr(s)}")
print(f"  Type: {type(s)}")
print(f"  Is valid Python string: {isinstance(s, str)}")

try:
    s.encode('utf-8')
    print("  UTF-8 encoding: SUCCESS")
except UnicodeEncodeError as e:
    print(f"  UTF-8 encoding: FAILED - {e}")

try:
    s.encode('utf-8', errors='surrogatepass')
    print("  UTF-8 with surrogatepass: SUCCESS")
except Exception as e:
    print(f"  UTF-8 with surrogatepass: FAILED - {e}")

try:
    encoded = s.encode('utf-8', errors='replace')
    print(f"  UTF-8 with replace: SUCCESS - {encoded}")
except Exception as e:
    print(f"  UTF-8 with replace: FAILED - {e}")

print("\n" + "=" * 60)
print("Testing the error handling in _hash_ndarray:")
print("-" * 40)

# Let's look at what the fallback does
print("\nThe code has a try/except that catches TypeError:")
print("  try:")
print("      vals = hash_object_array(vals, hash_key, encoding)")
print("  except TypeError:")
print("      vals = hash_object_array(vals.astype(str).astype(object), hash_key, encoding)")
print()
print("But UnicodeEncodeError is NOT caught!")
print("The TypeError fallback converts to string first, but that doesn't help with surrogates.")