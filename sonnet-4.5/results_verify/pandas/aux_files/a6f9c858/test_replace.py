#!/usr/bin/env python3
"""Test pandas StringMethods.replace() to compare with count()"""

import pandas as pd

print("Testing pandas StringMethods.replace() API")
print("=" * 60)

s = pd.Series(['test.test', 'hello(world', 'dot.here'])
print(f"Test Series: {s.tolist()}")
print()

# Test replace with regex=False
print("Test 1: replace('.', 'X', regex=False)")
result = s.str.replace('.', 'X', regex=False)
print(f"  Result: {result.tolist()}")

print("\nTest 2: replace('.', 'X', regex=True)")
result = s.str.replace('.', 'X', regex=True)
print(f"  Result: {result.tolist()}")

print("\nTest 3: replace('(', 'X', regex=False)")
result = s.str.replace('(', 'X', regex=False)
print(f"  Result: {result.tolist()}")

print("\nTest 4: replace('(', 'X', regex=True)")
try:
    result = s.str.replace('(', 'X', regex=True)
    print(f"  Result: {result.tolist()}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Check default value of regex parameter
import inspect
print("\n" + "=" * 60)
print("Checking StringMethods.replace() signature:")
sig = inspect.signature(pd.Series.str.replace)
print(f"  Parameters: {list(sig.parameters.keys())}")
for name, param in sig.parameters.items():
    if name == 'regex':
        print(f"  regex parameter default: {param.default}")