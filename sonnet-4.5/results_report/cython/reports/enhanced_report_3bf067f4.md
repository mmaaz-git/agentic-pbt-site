# Bug Report: Cython.Plex.Regexps.chars_to_ranges Incorrect Range Merging with Duplicate Characters

**Target**: `Cython.Plex.Regexps.chars_to_ranges`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `chars_to_ranges` function incorrectly merges character ranges when the input contains duplicate characters, causing it to include characters that are not present in the original input string.

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
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 18, in <module>
    test_chars_to_ranges_coverage()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 5, in test_chars_to_ranges_coverage
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 15, in test_chars_to_ranges_coverage
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

# Test with duplicate characters
s = '00'
ranges = chars_to_ranges(s)

covered = set()
for i in range(0, len(ranges), 2):
    code1, code2 = ranges[i], ranges[i+1]
    for code in range(code1, code2):
        covered.add(chr(code))

print(f"Input: {s!r}")
print(f"Ranges returned: {ranges}")
print(f"Expected coverage: {set(s)}")
print(f"Actual coverage: {covered}")
print(f"Extra characters incorrectly covered: {covered - set(s)}")

# Test with more duplicates
print("\n--- Test with more duplicates ---")
s2 = '0000000000'
ranges2 = chars_to_ranges(s2)

covered2 = set()
for i in range(0, len(ranges2), 2):
    code1, code2 = ranges2[i], ranges2[i+1]
    for code in range(code1, code2):
        covered2.add(chr(code))

print(f"Input: {s2!r}")
print(f"Ranges returned: {ranges2}")
print(f"Expected coverage: {set(s2)}")
print(f"Actual coverage: {covered2}")
print(f"Extra characters incorrectly covered: {covered2 - set(s2)}")

# Demonstrate impact on Any() and AnyBut()
print("\n--- Impact on Any() and AnyBut() ---")
from Cython.Plex import Lexicon, Scanner, Any, AnyBut, TEXT
from io import StringIO

# Any('00') should only match '0', but incorrectly matches '1' as well
print("Testing Any('00'):")
lexicon_any = Lexicon([(Any('00'), TEXT)])
scanner_any = Scanner(lexicon_any, StringIO('01'))
result = scanner_any.read()
if result:
    print(f"  Any('00') matched: {result[1]!r} (should only match '0')")

# AnyBut('00') should match '1', but raises UnrecognizedInput
print("\nTesting AnyBut('00'):")
try:
    lexicon_anybut = Lexicon([(AnyBut('00'), TEXT)])
    scanner_anybut = Scanner(lexicon_anybut, StringIO('1'))
    result = scanner_anybut.read()
    if result:
        print(f"  AnyBut('00') matched: {result[1]!r}")
except Exception as e:
    print(f"  AnyBut('00') raised error: {e.__class__.__name__}: {e}")
    print(f"  (This should NOT happen - '1' should be matched by AnyBut('00'))")
```

<details>

<summary>
chars_to_ranges incorrectly expands ranges when duplicates are present
</summary>
```
Input: '00'
Ranges returned: [48, 50]
Expected coverage: {'0'}
Actual coverage: {'1', '0'}
Extra characters incorrectly covered: {'1'}

--- Test with more duplicates ---
Input: '0000000000'
Ranges returned: [48, 58]
Expected coverage: {'0'}
Actual coverage: {'2', '0', '8', '7', '3', '9', '4', '6', '1', '5'}
Extra characters incorrectly covered: {'2', '8', '7', '3', '9', '4', '6', '1', '5'}

--- Impact on Any() and AnyBut() ---
Testing Any('00'):
  Any('00') matched: '0' (should only match '0')

Testing AnyBut('00'):
  AnyBut('00') raised error: UnrecognizedInput: '', line 0, char 0: Token not recognised in state ''
  (This should NOT happen - '1' should be matched by AnyBut('00'))
```
</details>

## Why This Is A Bug

The `chars_to_ranges` function's docstring clearly states it should "Return a list of character codes consisting of pairs [code1a, code1b, code2a, code2b,...] which cover all the characters in |s|". The function should handle duplicate characters by deduplicating them and covering only the unique characters present in the input.

However, the current implementation has a logic error on line 43 that causes incorrect range extension when processing duplicate characters:

```python
while i < n and code2 >= ord(char_list[i]):
    code2 += 1
    i += 1
```

When the input contains duplicates (e.g., '00'), after sorting, `char_list` becomes `['0', '0']`. When processing:
1. First '0': `code1 = 48`, `code2 = 49`, `i = 1`
2. The condition `code2 >= ord(char_list[i])` evaluates to `49 >= 48` (True)
3. This incorrectly increments `code2` to 50, extending the range to include character '1'
4. For each duplicate, the range expands by one additional character

This violates the function's contract and causes downstream issues:
- `Any('00')` matches both '0' and '1' instead of only '0'
- `AnyBut('00')` incorrectly excludes '1', causing `UnrecognizedInput` errors on valid input
- With many duplicates (e.g., '0000000000'), the range incorrectly expands to cover '0' through '9'

## Relevant Context

The `chars_to_ranges` function is a core utility in Cython's Plex lexical analyzer, used by the `Any()` and `AnyBut()` regular expression constructors. These constructors are fundamental building blocks for creating lexical patterns in Cython's parsing infrastructure.

The bug affects any pattern that contains duplicate characters, which can easily occur in real-world use cases:
- Character class definitions with accidental duplicates
- Programmatically generated patterns
- User-provided input patterns

The Plex module source code can be found at: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Plex/Regexps.py`

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

The fix changes the condition from `code2 >= ord(char_list[i])` to `code2 == ord(char_list[i])`. This ensures that the range only extends when the next character is exactly consecutive, correctly handling both duplicate characters and non-consecutive characters in the sorted list.