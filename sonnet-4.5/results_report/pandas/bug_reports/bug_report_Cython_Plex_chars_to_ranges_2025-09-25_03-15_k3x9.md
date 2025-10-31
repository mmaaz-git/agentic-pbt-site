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

**Failing input**: `s='00'`

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

print(f"Input string: {s!r}")
print(f"Unique characters in input: {set(s)}")
print(f"Ranges returned: {ranges}")
print(f"Characters covered by ranges: {covered}")
```

Output:
```
Input string: '00'
Unique characters in input: {'0'}
Ranges returned: [48, 50]
Characters covered by ranges: {'0', '1'}
```

## Why This Is A Bug

The function's docstring states it should "cover all the characters in |s|". For input `'00'`, the only unique character is `'0'` (ASCII 48). However, the function returns `[48, 50]`, which represents the range [48, 50) covering both `'0'` (48) and `'1'` (49).

The root cause is line 43 in `Regexps.py`. The condition `code2 >= ord(char_list[i])` incorrectly extends the range when encountering duplicate characters. When processing sorted `['0', '0']`:
1. First iteration: `code1 = 48`, `code2 = 49`, `i = 1`
2. Loop condition: `49 >= ord('0')` → `49 >= 48` → True
3. Extends range to `code2 = 50`, increments `i = 2`
4. Returns `[48, 50]` covering both '0' and '1'

This directly affects the `Any()` and `AnyBut()` regex constructors:

**Impact on `Any()`**: `Any('00')` incorrectly matches both '0' and '1', instead of only '0'.

**Impact on `AnyBut()`**: `AnyBut('00')` incorrectly rejects both '0' and '1', instead of only rejecting '0'. This causes the scanner to fail with `UnrecognizedInput` when scanning valid characters:

```python
from Cython.Plex import Lexicon, Scanner, AnyBut, TEXT
from io import StringIO

lexicon = Lexicon([(AnyBut('00'), TEXT)])
scanner = Scanner(lexicon, StringIO('1'))
scanner.read()
```

This raises `UnrecognizedInput` even though '1' should be matched by `AnyBut('00')`.

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

The fix changes the condition to check if the next character is exactly consecutive (`code2 == ord(char_list[i])`), rather than greater than or equal. This correctly handles duplicates by not extending the range.