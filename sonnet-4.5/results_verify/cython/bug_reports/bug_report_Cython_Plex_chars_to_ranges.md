# Bug Report: Cython.Plex.Regexps.chars_to_ranges Incorrect Range Merging

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly merges character ranges when the input contains duplicate characters, causing it to cover characters not present in the input string.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex.Regexps import chars_to_ranges

@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_chars_to_ranges_coverage(s):
    ranges = chars_to_ranges(s)

    covered_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i+1]
        for code in range(code1, code2):
            covered_chars.add(chr(code))

    assert set(s) == covered_chars
```

**Failing input**: `s='00'` (and many others with duplicates)

## Reproducing the Bug

```python
from Cython.Plex.Regexps import chars_to_ranges

s = '00'
ranges = chars_to_ranges(s)

covered = set()
for i in range(0, len(ranges), 2):
    code1, code2 = ranges[i], ranges[i+1]
    for code in range(code1, code2):
        covered.add(chr(code))

print(f"Input: {s!r}")
print(f"Expected coverage: {set(s)}")
print(f"Actual coverage: {covered}")
```

Output:
```
Input: '00'
Expected coverage: {'0'}
Actual coverage: {'0', '1'}
```

For more extreme cases with many duplicates:
```python
s = '0000000000'
ranges = chars_to_ranges(s)
```
This incorrectly returns ranges covering '0' through '9'.

## Why This Is A Bug

The function's docstring states it should "cover all the characters in |s|". For input with duplicates, it should deduplicate and cover only the unique characters. However, due to an incorrect condition on line 43, it extends ranges beyond the input characters.

**Root cause** (Regexps.py:43):
```python
while i < n and code2 >= ord(char_list[i]):
```

When processing duplicates, this condition is always True (e.g., for '00': `49 >= 48`), causing the range to extend by 1 for each duplicate character.

**Impact**: This affects `Any()` and `AnyBut()` regex constructors:
- `Any('00')` incorrectly matches both '0' and '1' instead of only '0'
- `AnyBut('00')` incorrectly rejects '1', causing `UnrecognizedInput` errors:

```python
from Cython.Plex import Lexicon, Scanner, AnyBut, TEXT
from io import StringIO

lexicon = Lexicon([(AnyBut('00'), TEXT)])
scanner = Scanner(lexicon, StringIO('1'))
scanner.read()
```
This raises `UnrecognizedInput` even though '1' should match.

## Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -40,7 +40,7 @@ def chars_to_ranges(s):
         code1 = ord(char_list[i])
         code2 = code1 + 1
         i += 1
-        while i < n and code2 >= ord(char_list[i]):
+        while i < n and code2 == ord(char_list[i]):
             code2 += 1
             i += 1
         result.append(code1)
```

The fix checks if the next character is exactly consecutive (`code2 == ord(char_list[i])`), properly handling duplicates and non-consecutive characters.