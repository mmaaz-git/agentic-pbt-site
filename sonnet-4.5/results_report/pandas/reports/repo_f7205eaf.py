#!/usr/bin/env python3
"""
Minimal reproduction of the pandas slice_replace bug
"""
import pandas as pd

print("Testing pandas.core.strings.slice_replace bug")
print("=" * 60)

# Test case 1: Simple case with start > stop
print("\nTest case 1: strings=['0'], start=1, stop=0, repl=''")
s = pd.Series(['0'])
result = s.str.slice_replace(start=1, stop=0, repl='')

print(f"Pandas result:   {result.iloc[0]!r}")
print(f"Expected result: {'0'[:1] + '' + '0'[0:]!r}")
print(f"Are they equal?  {result.iloc[0] == ('0'[:1] + '' + '0'[0:])}")

# Verify Python slicing behavior
print(f"\nPython slicing verification:")
print(f"'0'[1:0] = {'0'[1:0]!r} (empty string)")
print(f"'0'[:1] = {'0'[:1]!r}")
print(f"'0'[0:] = {'0'[0:]!r}")
print(f"'0'[:1] + '' + '0'[0:] = {'0'[:1] + '' + '0'[0:]!r}")

print("\n" + "-" * 60)

# Test case 2: More complex case with replacement string
print("\nTest case 2: strings=['hello'], start=3, stop=1, repl='X'")
s2 = pd.Series(['hello'])
result2 = s2.str.slice_replace(start=3, stop=1, repl='X')

print(f"Pandas result:   {result2.iloc[0]!r}")
print(f"Expected result: {'hello'[:3] + 'X' + 'hello'[1:]!r}")
print(f"Are they equal?  {result2.iloc[0] == ('hello'[:3] + 'X' + 'hello'[1:])}")

# Verify Python slicing behavior
print(f"\nPython slicing verification:")
print(f"'hello'[3:1] = {'hello'[3:1]!r} (empty string)")
print(f"'hello'[:3] = {'hello'[:3]!r}")
print(f"'hello'[1:] = {'hello'[1:]!r}")
print(f"'hello'[:3] + 'X' + 'hello'[1:] = {'hello'[:3] + 'X' + 'hello'[1:]!r}")

print("\n" + "=" * 60)
print("Summary: The bug occurs when start > stop.")
print("Pandas incorrectly adjusts the stop position, producing wrong results.")