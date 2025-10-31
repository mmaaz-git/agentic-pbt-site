# Bug Report: Cython.Plex.Regexps chars_to_ranges Duplicate Character Handling

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly expands character ranges when the input string contains duplicate characters, causing `Any()` and `AnyBut()` patterns to match characters that were not in the original string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Plex.Regexps import chars_to_ranges

@given(st.text())
def test_chars_to_ranges_preserves_all_characters(s):
    ranges = chars_to_ranges(s)

    assert len(ranges) % 2 == 0

    reconstructed_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i + 1]
        for code in range(code1, code2):
            reconstructed_chars.add(chr(code))

    assert reconstructed_chars == set(s)
```

**Failing input**: `s='00'`

## Reproducing the Bug

Low-level reproduction:

```python
from Cython.Plex.Regexps import chars_to_ranges

s = '00'
ranges = chars_to_ranges(s)

reconstructed = set()
for i in range(0, len(ranges), 2):
    code1, code2 = ranges[i], ranges[i + 1]
    for code in range(code1, code2):
        reconstructed.add(chr(code))

assert reconstructed == {'0', '1'}, f"Expected {{'0'}}, got {reconstructed}"
```

High-level impact demonstration (affects `Any()` public API):

```python
from io import StringIO
from Cython.Plex import Any, Lexicon, Scanner, TEXT

lexicon = Lexicon([(Any('00'), TEXT)])

scanner = Scanner(lexicon, StringIO('1'))
value, text = scanner.read()
assert text == '1'
```

## Why This Is A Bug

The function's docstring states it should "Return a list of character codes... which cover all the characters in |s|." When given '00', the set of characters is {'0'}, but the function returns ranges [48, 50] which covers both '0' (48) and '1' (49). This violates the documented contract.

The bug occurs because the function uses `>=` instead of `>` when checking if consecutive characters should be merged into a range. When processing duplicate characters like '00', after sorting we have ['0', '0']. The code sets `code2 = code1 + 1 = 49`, then checks if `49 >= ord('0')` (which is 48), which is true, so it incorrectly extends the range and increments both `code2` and `i`, skipping the duplicate.

This affects the public `Any()` function, causing `Any('00')` to match both '0' and '1'.

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