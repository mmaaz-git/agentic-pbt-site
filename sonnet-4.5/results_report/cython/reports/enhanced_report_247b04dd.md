# Bug Report: Cython.Plex.Regexps chars_to_ranges Incorrectly Extends Ranges for Duplicate Characters

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly extends character ranges when processing duplicate characters, causing it to include characters that were not in the original input string.

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


if __name__ == "__main__":
    test_chars_to_ranges_coverage()
```

<details>

<summary>
**Failing input**: `'00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 21, in <module>
    test_chars_to_ranges_coverage()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 6, in test_chars_to_ranges_coverage
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 17, in test_chars_to_ranges_coverage
    assert covered_chars == original_chars
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_chars_to_ranges_coverage(
    s='00',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py:44
```
</details>

## Reproducing the Bug

```python
from Cython.Plex.Regexps import chars_to_ranges

# Test with duplicate characters
test_cases = ['00', 'aaa', 'aabbcc']

for s in test_cases:
    print(f"\n=== Testing input: '{s}' ===")
    ranges = chars_to_ranges(s)
    print(f"Returned ranges: {ranges}")

    # Decode what characters are covered by these ranges
    covered_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i + 1]
        print(f"Range [{code1}, {code2}): ", end="")
        chars_in_range = []
        for code in range(code1, code2):
            char = chr(code)
            covered_chars.add(char)
            chars_in_range.append(f"'{char}'")
        print(", ".join(chars_in_range))

    original_chars = set(s)
    print(f"Original characters: {original_chars}")
    print(f"Covered characters: {covered_chars}")

    if covered_chars == original_chars:
        print("✓ CORRECT: Ranges cover exactly the input characters")
    else:
        extra = covered_chars - original_chars
        missing = original_chars - covered_chars
        if extra:
            print(f"✗ BUG: Ranges include extra characters: {extra}")
        if missing:
            print(f"✗ BUG: Ranges missing characters: {missing}")
```

<details>

<summary>
Bug demonstration showing incorrect range extension
</summary>
```

=== Testing input: '00' ===
Returned ranges: [48, 50]
Range [48, 50): '0', '1'
Original characters: {'0'}
Covered characters: {'0', '1'}
✗ BUG: Ranges include extra characters: {'1'}

=== Testing input: 'aaa' ===
Returned ranges: [97, 100]
Range [97, 100): 'a', 'b', 'c'
Original characters: {'a'}
Covered characters: {'c', 'a', 'b'}
✗ BUG: Ranges include extra characters: {'c', 'b'}

=== Testing input: 'aabbcc' ===
Returned ranges: [97, 103]
Range [97, 103): 'a', 'b', 'c', 'd', 'e', 'f'
Original characters: {'c', 'a', 'b'}
Covered characters: {'f', 'e', 'c', 'b', 'a', 'd'}
✗ BUG: Ranges include extra characters: {'e', 'f', 'd'}
```
</details>

## Why This Is A Bug

The `chars_to_ranges` function's docstring explicitly states it should return ranges that "cover all the characters in |s|". The natural interpretation is that it should cover exactly those characters, not additional ones. When the function receives duplicate characters, it incorrectly extends the range to include characters between duplicates that were never in the input.

This bug propagates to the public `Any()` constructor, which is documented as "an RE which matches any character in the string |s|". When `Any('00')` matches both '0' and '1', this violates the documented behavior - it should only match characters that are literally "in the string".

The bug occurs at line 43 of Regexps.py where the condition `code2 >= ord(char_list[i])` incorrectly increments `code2` when encountering duplicate characters. For example, when processing '00', after handling the first '0' (code=48), `code2` becomes 49. When checking the second '0', the condition `49 >= 48` is true, causing `code2` to increment to 50, thus including character '1' (code 49) in the range.

## Relevant Context

The `chars_to_ranges` function is used internally by several RE constructors in Cython's Plex module:
- `Any(s)` at line 436: Creates an RE matching any character in string s
- `AnyBut(s)` at line 446: Creates an RE matching any character NOT in string s

Both of these public APIs would be affected by this bug. The issue only manifests when the input string contains duplicate characters. The function attempts to handle duplicates by sorting the input (line 35), but the range-building logic incorrectly extends ranges when duplicates are present.

The algorithm appears to be attempting to build minimal ranges by extending them when consecutive characters are found, but it fails to account for the case where the "next" character is actually a duplicate of one already included in the range.

## Proposed Fix

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