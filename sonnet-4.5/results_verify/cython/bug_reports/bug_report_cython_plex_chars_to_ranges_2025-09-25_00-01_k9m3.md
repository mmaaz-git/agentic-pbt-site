# Bug Report: Cython.Plex.Regexps chars_to_ranges Incorrect Handling of Duplicates

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly extends character ranges when encountering duplicate characters, causing the `Any()` regular expression constructor to match characters that were not in the input set.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex.Regexps import chars_to_ranges


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=1000)
def test_chars_to_ranges_coverage(s):
    ranges = chars_to_ranges(s)

    covered_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i + 1]
        for code in range(code1, code2):
            covered_chars.add(chr(code))

    original_chars = set(s)
    assert covered_chars == original_chars
```

**Failing input**: `'00'`, `'aaa'`, `'aabbcc'`

## Reproducing the Bug

```python
from Cython.Plex.Regexps import chars_to_ranges

ranges = chars_to_ranges('00')
print(ranges)

covered = set()
for i in range(0, len(ranges), 2):
    for code in range(ranges[i], ranges[i+1]):
        covered.add(chr(code))

print(f"Input: {set('00')}")
print(f"Covered: {covered}")
```

Output:
```
[48, 50]
Input: {'0'}
Covered: {'0', '1'}
```

The function incorrectly includes '1' in the range when only '0' was in the input.

## Why This Is A Bug

The docstring states the function should "cover all the characters in |s|" - not extra characters. When duplicate characters appear in the input, the function incorrectly extends the range to include characters beyond those in the input string.

This affects `Any(s)`, which is documented to match "any character in the string |s|". For example, `Any('00')` will incorrectly match both '0' and '1'.

## Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -40,7 +40,7 @@ def chars_to_ranges(s):
         code1 = ord(char_list[i])
         code2 = code1 + 1
         i += 1
-        while i < n and code2 >= ord(char_list[i]):
+        while i < n and code2 > ord(char_list[i]):
             code2 += 1
             i += 1
         result.append(code1)
```

The condition should be `>` instead of `>=` to correctly skip duplicate characters without extending the range.