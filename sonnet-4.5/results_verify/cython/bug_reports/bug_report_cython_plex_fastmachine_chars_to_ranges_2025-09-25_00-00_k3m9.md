# Bug Report: Cython.Plex.Machines FastMachine.chars_to_ranges Duplicate Characters

**Target**: `Cython.Plex.Machines.FastMachine.chars_to_ranges`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` method in `FastMachine` creates duplicate ranges when the input character list contains duplicate characters, instead of producing a minimal set of non-overlapping ranges.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Plex.Machines import FastMachine

@given(st.lists(st.characters(), min_size=1, max_size=50))
def test_chars_to_ranges_consecutive_merging(char_list):
    fm = FastMachine()
    ranges = fm.chars_to_ranges(char_list)

    for i in range(len(ranges) - 1):
        c1_end = ord(ranges[i][1])
        c2_start = ord(ranges[i + 1][0])
        assert c1_end + 1 < c2_start
```

**Failing input**: `['0', '0']`

## Reproducing the Bug

```python
from Cython.Plex.Machines import FastMachine

fm = FastMachine()

result = fm.chars_to_ranges(['0', '0'])

print(f"Result: {result}")

print(f"Unique characters: {set(['0', '0'])}")
assert len(result) == 1, f"Expected 1 range for 1 unique character, got {len(result)}"
```

## Why This Is A Bug

The `chars_to_ranges` method is used by `dump_transitions` (line 201) to create a compact representation of character transitions for debugging output. When duplicate characters are present in the input, the method creates duplicate ranges in the output, which is both inefficient and produces misleading debugging output.

The bug occurs because the inner while loop condition on line 224 only checks for consecutive characters (`ord(char_list[i]) == c2 + 1`) but doesn't handle duplicate characters (`ord(char_list[i]) == c2`). This causes each duplicate to start a new range instead of being merged into the existing one.

## Fix

```diff
--- a/Cython/Plex/Machines.py
+++ b/Cython/Plex/Machines.py
@@ -221,7 +221,7 @@ class FastMachine:
             c1 = ord(char_list[i])
             c2 = c1
             i += 1
-            while i < n and ord(char_list[i]) == c2 + 1:
+            while i < n and ord(char_list[i]) <= c2 + 1:
                 i += 1
                 c2 += 1
             result.append((chr(c1), chr(c2)))
```

The fix changes the condition from `== c2 + 1` to `<= c2 + 1`, which handles both duplicate characters (when `ord(char_list[i]) == c2`) and consecutive characters (when `ord(char_list[i]) == c2 + 1`).