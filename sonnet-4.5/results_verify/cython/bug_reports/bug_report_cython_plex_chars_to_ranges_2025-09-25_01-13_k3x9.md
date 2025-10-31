# Bug Report: Cython.Plex chars_to_ranges Incorrect Range on Duplicates

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly expands character ranges when the input string contains duplicate characters, covering characters that are not in the input. This affects `Any()` and `AnyBut()` regular expression constructors.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from Cython.Plex.Regexps import chars_to_ranges


@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_chars_to_ranges_coverage(s):
    result = chars_to_ranges(s)

    covered_chars = set()
    for i in range(0, len(result), 2):
        code1 = result[i]
        code2 = result[i + 1]
        for code in range(code1, code2):
            covered_chars.add(chr(code))

    input_chars = set(s)
    assert covered_chars == input_chars
```

**Failing input**: `s='00'`

## Reproducing the Bug

```python
from Cython.Plex.Regexps import chars_to_ranges

s = '00'
result = chars_to_ranges(s)

covered = set()
for i in range(0, len(result), 2):
    for code in range(result[i], result[i + 1]):
        covered.add(chr(code))

assert covered == {'0', '1'}
assert set(s) == {'0'}
```

## Why This Is A Bug

The function's docstring claims it returns ranges that "cover all the characters in |s|". However, when the input contains duplicate characters (e.g., '00'), the function produces ranges that cover additional characters not present in the input. In this case, inputting '00' produces ranges `[48, 50]` which covers both '0' and '1', even though '1' is not in the input string.

The root cause is in line 43 of `/Cython/Plex/Regexps.py`:
```python
while i < n and code2 >= ord(char_list[i]):
```

When processing duplicates, `code2` (which is `ord(current_char) + 1`) is greater than or equal to `ord(duplicate_char)`, causing the range to incorrectly expand.

This bug propagates to `Any()` and `AnyBut()` functions which use `chars_to_ranges`, potentially causing incorrect pattern matching in lexical scanners.

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

The fix changes `>=` to `>`, which prevents the range from expanding when encountering duplicate characters, while still correctly merging consecutive character codes.