# Bug Report: Cython.Plex.Regexps.chars_to_ranges Incorrect Range Merging with Duplicate Characters

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly merges character ranges when the input contains duplicate characters, causing it to include characters that were not present in the original input string.

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

if __name__ == "__main__":
    test_chars_to_ranges_coverage()
```

<details>

<summary>
**Failing input**: `s='00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 18, in <module>
    test_chars_to_ranges_coverage()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 5, in test_chars_to_ranges_coverage
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 15, in test_chars_to_ranges_coverage
    assert set(s) == covered_chars
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_chars_to_ranges_coverage(
    s='00',
)
```
</details>

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

<details>

<summary>
Characters covered incorrectly include '1' despite only '0' being in input
</summary>
```
Input string: '00'
Unique characters in input: {'0'}
Ranges returned: [48, 50]
Characters covered by ranges: {'1', '0'}
```
</details>

## Why This Is A Bug

The function's docstring states it should return "A list of character codes suitable for use in Transitions.TransitionMap" that "cover all the characters in |s|". For the input `'00'`, the only unique character is `'0'` (ASCII code 48). However, the function returns the range `[48, 50]`, which represents the half-open interval [48, 50) covering both `'0'` (48) and `'1'` (49). This violates the function's contract by including a character ('1') that was never present in the input string.

The bug occurs at line 43 of the implementation. When processing the sorted character list `['0', '0']`:
1. First iteration sets `code1 = 48`, `code2 = 49`, and advances `i = 1`
2. The while loop condition `code2 >= ord(char_list[i])` evaluates to `49 >= ord('0')` → `49 >= 48` → True
3. This incorrectly extends `code2` to 50 and increments `i` to 2
4. The resulting range `[48, 50]` incorrectly covers both '0' and '1'

This bug critically impacts the `Any()` and `AnyBut()` regex constructors that rely on `chars_to_ranges`:
- `Any('00')` will incorrectly match the character '1' in addition to '0'
- `AnyBut('00')` will incorrectly reject both '0' and '1', causing `UnrecognizedInput` exceptions when scanning the character '1' which should be valid

## Relevant Context

The `chars_to_ranges` function is located in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Plex/Regexps.py` at lines 28-48. It's a core utility function used by the Cython lexical analyzer's regex engine.

The function is designed to convert a string of characters into a compact list of character code ranges for efficient character class representation. The ranges are represented as pairs `[start, end)` where `start` is inclusive and `end` is exclusive.

Key usage points:
- `Any(chars)` constructor (line 516): Creates a regex matching any character in the input
- `AnyBut(chars)` constructor (line 526): Creates a regex matching any character NOT in the input
- Both are fundamental building blocks for Cython's lexical analysis

Documentation reference: The function docstring at line 29-35 explicitly states the ranges should "cover all the characters in |s|" - meaning exactly those characters, not additional ones.

## Proposed Fix

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