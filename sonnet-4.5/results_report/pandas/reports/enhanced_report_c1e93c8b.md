# Bug Report: Cython.Plex.Regexps.Range IndexError on Odd-Length String Input

**Target**: `Cython.Plex.Regexps.Range`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Range` function in Cython's Plex module crashes with an uninformative `IndexError` when provided an odd-length string, instead of validating the input and raising a descriptive error about the even-length requirement documented in its docstring.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex.Regexps import Range

@given(st.text(min_size=1).filter(lambda s: len(s) % 2 == 1))
@settings(max_examples=200)
def test_range_validates_even_length(s):
    Range(s)

if __name__ == "__main__":
    test_range_validates_even_length()
```

<details>

<summary>
**Failing input**: `s='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 10, in <module>
    test_range_validates_even_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 5, in test_range_validates_even_length
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 7, in test_range_validates_even_length
    Range(s)
    ~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py", line 475, in Range
    ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
                                            ~~^^^^^^^
IndexError: string index out of range
Falsifying example: test_range_validates_even_length(
    s='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from Cython.Plex.Regexps import Range

# Test with odd-length string that should cause IndexError
s = 'abc'
result = Range(s)
print(f"Range('{s}') succeeded: {result}")
```

<details>

<summary>
IndexError when accessing string index out of range
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/repo.py", line 5, in <module>
    result = Range(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Regexps.py", line 475, in Range
    ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
                                            ~~^^^^^^^
IndexError: string index out of range
```
</details>

## Why This Is A Bug

The `Range` function's docstring explicitly states: "Range(s) where |s| is a string of even length is an RE which matches any single character in the ranges |s[0]| to |s[1]|, |s[2]| to |s[3]|,..." This establishes a clear precondition that single-argument strings must have even length.

However, the function fails to validate this documented requirement. When an odd-length string is provided, the code at line 475 attempts to access `s1[i + 1]` in the final iteration of the loop `for i in range(0, len(s1), 2)`. For odd-length strings, this results in accessing an index beyond the string bounds, causing an `IndexError`.

This violates the principle of failing fast with meaningful error messages. Users receive a generic "string index out of range" error that provides no context about the actual problem - that the string must have even length. The error message exposes implementation details rather than explaining the contract violation, making debugging unnecessarily difficult.

## Relevant Context

The `Range` function is part of Cython's Plex module, which is used for lexical analysis and regular expression building. The function has two forms:

1. **Two-argument form** `Range(c1, c2)`: Creates a regex matching characters from c1 to c2 inclusive - works correctly
2. **Single-argument form** `Range(s)`: Creates multiple character ranges by pairing adjacent characters in the string - crashes on odd-length input

The bug occurs in the single-argument form at line 475 of `/Cython/Plex/Regexps.py`. The loop iterates with step 2 through the string, attempting to create character ranges from pairs like `(s[0], s[1])`, `(s[2], s[3])`, etc. When the string has odd length, the final character has no pair, causing the index error.

While this is an internal module primarily used by Cython's lexical analyzer, the function is importable and callable by users, making it part of the de facto public API. The fix would improve debugging experience without any performance impact.

## Proposed Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -471,6 +471,8 @@ def Range(s1, s2=None):
         result.str = "Range(%s,%s)" % (s1, s2)
     else:
+        if len(s1) % 2 != 0:
+            raise PlexValueError("Range() requires a string of even length, got length %d" % len(s1))
         ranges = []
         for i in range(0, len(s1), 2):
             ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
```