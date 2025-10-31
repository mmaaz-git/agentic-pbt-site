# Bug Report: Cython.Plex.Regexps chars_to_ranges Incorrect Range Expansion with Duplicate Characters

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly expands character ranges when the input string contains duplicate characters, causing it to include additional characters that were not present in the original input string.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis property test for chars_to_ranges function."""

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

if __name__ == "__main__":
    # Run the property test
    test_chars_to_ranges_preserves_all_characters()
```

<details>

<summary>
**Failing input**: `s='00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 23, in <module>
    test_chars_to_ranges_preserves_all_characters()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 8, in test_chars_to_ranges_preserves_all_characters
    def test_chars_to_ranges_preserves_all_characters(s):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 19, in test_chars_to_ranges_preserves_all_characters
    assert reconstructed_chars == set(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_chars_to_ranges_preserves_all_characters(
    s='00',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py:44
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstrating the chars_to_ranges bug with duplicate characters."""

from Cython.Plex.Regexps import chars_to_ranges
from io import StringIO
from Cython.Plex import Any, Lexicon, Scanner, TEXT

# Low-level demonstration of the bug
print("=== Low-level demonstration ===")
print("Testing chars_to_ranges('00')...")

s = '00'
ranges = chars_to_ranges(s)
print(f"Input string: {repr(s)}")
print(f"Returned ranges: {ranges}")

# Reconstruct characters from ranges
reconstructed = set()
for i in range(0, len(ranges), 2):
    code1, code2 = ranges[i], ranges[i + 1]
    for code in range(code1, code2):
        reconstructed.add(chr(code))

print(f"Reconstructed characters: {reconstructed}")
print(f"Expected characters: {set(s)}")

# Check if they match
if reconstructed != set(s):
    print(f"ERROR: Expected {set(s)}, but got {reconstructed}")
    print(f"The function incorrectly includes character(s): {reconstructed - set(s)}")
else:
    print("OK: Reconstructed characters match input")

print()

# High-level demonstration showing impact on public API
print("=== High-level impact on Any() function ===")
print("Creating lexicon with Any('00')...")

lexicon = Lexicon([(Any('00'), TEXT)])
print("Any('00') should match only '0', but let's test if it matches '1'...")

scanner = Scanner(lexicon, StringIO('1'))
value, text = scanner.read()

if text == '1':
    print(f"ERROR: Any('00') incorrectly matched '1'")
    print(f"This demonstrates that the bug affects the public API")
else:
    print(f"Any('00') did not match '1' (unexpected)")

print()
print("=== Explanation ===")
print("The bug occurs because chars_to_ranges uses '>=' instead of '>' when")
print("checking if consecutive characters should be merged into a range.")
print("With duplicate '0' characters, it incorrectly expands the range to include '1'.")
```

<details>

<summary>
ERROR: chars_to_ranges incorrectly expands range to include extra character
</summary>
```
=== Low-level demonstration ===
Testing chars_to_ranges('00')...
Input string: '00'
Returned ranges: [48, 50]
Reconstructed characters: {'1', '0'}
Expected characters: {'0'}
ERROR: Expected {'0'}, but got {'1', '0'}
The function incorrectly includes character(s): {'1'}

=== High-level impact on Any() function ===
Creating lexicon with Any('00')...
Any('00') should match only '0', but let's test if it matches '1'...
ERROR: Any('00') incorrectly matched '1'
This demonstrates that the bug affects the public API

=== Explanation ===
The bug occurs because chars_to_ranges uses '>=' instead of '>' when
checking if consecutive characters should be merged into a range.
With duplicate '0' characters, it incorrectly expands the range to include '1'.
```
</details>

## Why This Is A Bug

The `chars_to_ranges` function explicitly documents that it should "Return a list of character codes... which cover all the characters in |s|" (lines 30-32 in Regexps.py). When given the input string '00', the set of unique characters is {'0'}, but the function returns ranges [48, 50] which represents the half-open interval covering both '0' (ASCII 48) and '1' (ASCII 49).

This violates the documented contract in multiple ways:

1. **Documentation violation**: The function promises to cover "all the characters in |s|" - not more, not less. Including '1' when it was never in the input string directly contradicts this specification.

2. **API impact**: The bug propagates to the public `Any()` function (line 436) which states it "matches any character in the string |s|". When `Any('00')` matches '1', it breaks this promise. Similarly, `AnyBut()` would also be affected.

3. **Root cause**: The bug is on line 43 of the `chars_to_ranges` function. When processing sorted duplicates like ['0', '0'], the condition `code2 >= ord(char_list[i])` evaluates to `49 >= 48` (True) for the second '0'. This causes the algorithm to incorrectly extend the range and skip over the duplicate, resulting in a range that includes unintended characters.

4. **Lexical analysis impact**: Cython.Plex is used for building lexical analyzers. Incorrect character matching could cause parsers to accept invalid tokens or reject valid ones, leading to subtle bugs in compilers and parsers built with this library.

## Relevant Context

- The Cython.Plex module is part of Cython, a compiler for Python extensions
- This module implements a lexical analyzer generator similar to Lex/Flex
- The `chars_to_ranges` function is a core utility used by pattern matching functions
- Character ranges are fundamental to regular expression matching in lexical analysis
- Bug affects all versions where line 43 uses `>=` instead of `>` comparison
- Workaround: Users can deduplicate input strings before passing to `Any()` or `AnyBut()`

Code location: `/Cython/Plex/Regexps.py` lines 28-48
Documentation: https://cython.readthedocs.io/

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