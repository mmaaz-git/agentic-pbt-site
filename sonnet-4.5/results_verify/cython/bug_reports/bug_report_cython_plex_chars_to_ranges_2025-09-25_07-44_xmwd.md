# Bug Report: Cython.Plex.Regexps chars_to_ranges Duplicate Character Handling

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly extends character ranges when duplicate characters are present, causing `Any()` regex patterns to match characters that were not specified.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=100))
def test_chars_to_ranges_roundtrip(s):
    from Cython.Plex.Regexps import chars_to_ranges
    ranges = chars_to_ranges(s)
    assert len(ranges) % 2 == 0
    reconstructed_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i+1]
        for code in range(code1, code2):
            reconstructed_chars.add(chr(code))
    assert reconstructed_chars == set(s)
```

**Failing input**: `'aa'`

## Reproducing the Bug

```python
from Cython.Plex.Regexps import chars_to_ranges
import Cython.Plex as plex
from io import StringIO

result = chars_to_ranges('aa')
chars_from_result = set(chr(c) for c in range(result[0], result[1]))
print(f"chars_to_ranges('aa') = {result}")
print(f"Expected: {{'a'}}, Got: {chars_from_result}")

lexicon = plex.Lexicon([
    (plex.Any('aa'), 'MATCH_A'),
    (plex.AnyChar, 'OTHER')
])

for char in ['a', 'b']:
    scanner = plex.Scanner(lexicon, StringIO(char))
    token, value = scanner.read()
    print(f"Input '{char}': token={token}")
    if char == 'b':
        print(f"  ERROR: Any('aa') incorrectly matched 'b'")
```

**Output:**
```
chars_to_ranges('aa') = [97, 99]
Expected: {'a'}, Got: {'a', 'b'}
Input 'a': token=MATCH_A
Input 'b': token=MATCH_A
  ERROR: Any('aa') incorrectly matched 'b'
```

## Why This Is A Bug

The function's purpose is to convert a string of characters into a compact list of character code ranges. When given `'aa'`, it should recognize that both characters are identical and produce a range covering only 'a' (codes [97, 98)). Instead, it incorrectly produces [97, 99), which includes both 'a' and 'b'.

This violates the documented behavior of `Any(s)` which states it "matches any character in the string |s|". With the bug, `Any('aa')` matches both 'a' and 'b', even though 'b' is not in the input string.

## Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -15,7 +15,7 @@ def chars_to_ranges(s):
         code1 = ord(char_list[i])
         code2 = code1 + 1
         i += 1
-        while i < n and code2 >= ord(char_list[i]):
+        while i < n and code2 > ord(char_list[i]):
             code2 += 1
             i += 1
         result.append(code1)
```

The condition should be `code2 > ord(char_list[i])` instead of `code2 >= ord(char_list[i])`. This ensures that duplicate characters don't incorrectly extend the range, while still properly merging consecutive characters like 'abc' into a single range.