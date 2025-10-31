#!/usr/bin/env python3
"""Test the reported pandas Series.str.slice_replace bug"""

import pandas as pd

# Test Case 1: The main reported bug
print("Test Case 1: strings=['abc'], start=2, stop=1")
s = pd.Series(['abc'])
result = s.str.slice_replace(start=2, stop=1, repl='X').iloc[0]

# Manual calculation of expected value
# When start > stop, Python slice 'abc'[2:1] returns ''
# So we're replacing an empty string at position 2
# Expected: 'abc'[:2] + 'X' + 'abc'[1:] = 'ab' + 'X' + 'bc' = 'abXbc'
expected = 'abc'[:2] + 'X' + 'abc'[1:]

print(f"  Original:  {s.iloc[0]!r}")
print(f"  Result:    {result!r}")
print(f"  Expected:  {expected!r}")
print(f"  Match:     {result == expected}")
print()

# Test Case 2: The negative index example
print("Test Case 2: strings=['hello'], start=-1, stop=-3")
s2 = pd.Series(['hello'])
result2 = s2.str.slice_replace(-1, -3, 'X').iloc[0]

# Manual calculation: 'hello'[-1:-3] = '' (empty slice)
# Expected: 'hello'[:-1] + 'X' + 'hello'[-3:] = 'hell' + 'X' + 'llo' = 'hellXllo'
expected2 = 'hello'[:-1] + 'X' + 'hello'[-3:]

print(f"  Original:  {s2.iloc[0]!r}")
print(f"  Result:    {result2!r}")
print(f"  Expected:  {expected2!r}")
print(f"  Match:     {result2 == expected2}")
print()

# Test Case 3: Let's verify what Python slicing actually does
print("Test Case 3: Python slicing behavior verification")
test_str = "abc"
print(f"  'abc'[2:1] = {test_str[2:1]!r} (empty string)")
print(f"  'abc'[:2] = {test_str[:2]!r}")
print(f"  'abc'[1:] = {test_str[1:]!r}")
print(f"  'abc'[2:] = {test_str[2:]!r}")
print()

# Test Case 4: More edge cases
print("Test Case 4: More edge cases with start > stop")
s3 = pd.Series(['0123456789'])
# start=5, stop=3 should preserve everything except empty slice at position 5
result3 = s3.str.slice_replace(start=5, stop=3, repl='X').iloc[0]
expected3 = '0123456789'[:5] + 'X' + '0123456789'[3:]
print(f"  Original:  {s3.iloc[0]!r}")
print(f"  Result:    {result3!r}")
print(f"  Expected:  {expected3!r}")
print(f"  Match:     {result3 == expected3}")

# Test Case 5: Normal case (start < stop) for comparison
print("\nTest Case 5: Normal case with start < stop (for comparison)")
s4 = pd.Series(['abc'])
result4 = s4.str.slice_replace(start=1, stop=2, repl='X').iloc[0]
expected4 = 'abc'[:1] + 'X' + 'abc'[2:]
print(f"  Original:  {s4.iloc[0]!r}")
print(f"  Result:    {result4!r}")
print(f"  Expected:  {expected4!r}")
print(f"  Match:     {result4 == expected4}")