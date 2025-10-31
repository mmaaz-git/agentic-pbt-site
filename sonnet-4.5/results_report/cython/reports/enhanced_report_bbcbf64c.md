# Bug Report: Cython.Utils.build_hex_version Crashes on Valid PEP 440 Pre-release Versions Without Explicit Numbers

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `build_hex_version` function crashes with `ValueError` when processing valid PEP 440 version strings that have pre-release tags (a, b, rc) without an explicit number suffix, which should implicitly default to 0 according to PEP 440.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utils import build_hex_version
import re

@settings(max_examples=500)
@given(st.from_regex(r'^[0-9]+\.[0-9]+(\.[0-9]+)?([ab]|rc)?[0-9]*$', fullmatch=True))
def test_build_hex_version_format(version_string):
    result = build_hex_version(version_string)
    assert re.match(r'^0x[0-9A-F]{8}$', result)

# Run the test
if __name__ == "__main__":
    test_build_hex_version_format()
```

<details>

<summary>
**Failing input**: `'0.0rc'`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 13, in <module>
  |     test_build_hex_version_format()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 6, in test_build_hex_version_format
  |     @given(st.from_regex(r'^[0-9]+\.[0-9]+(\.[0-9]+)?([ab]|rc)?[0-9]*$', fullmatch=True))
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 9, in test_build_hex_version_format
    |     assert re.match(r'^0x[0-9A-F]{8}$', result)
    |            ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_build_hex_version_format(
    |     version_string='300.0',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 8, in test_build_hex_version_format
    |     result = build_hex_version(version_string)
    |   File "Cython/Utils.py", line 611, in Cython.Utils.build_hex_version
    | ValueError: invalid literal for int() with base 10: ''
    | Falsifying example: test_build_hex_version_format(
    |     version_string='0.0rc',
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from Cython.Utils import build_hex_version

# Test case that crashes
version = '0.0rc'
print(f"Testing build_hex_version('{version}')")
result = build_hex_version(version)
print(f"Result: {result}")
```

<details>

<summary>
ValueError: invalid literal for int() with base 10: ''
</summary>
```
Testing build_hex_version('0.0rc')
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/repo.py", line 6, in <module>
    result = build_hex_version(version)
  File "Cython/Utils.py", line 611, in Cython.Utils.build_hex_version
ValueError: invalid literal for int() with base 10: ''
```
</details>

## Why This Is A Bug

The function's docstring explicitly references PEP 440 as the specification it follows for public version identifiers. According to PEP 440, pre-release versions can omit the numeral, in which case it is implicitly assumed to be `0`. The PEP states: "Pre releases allow omitting the numeral in which case it is implicitly assumed to be `0`. The normal form for this is to include the `0` explicitly."

This means version strings like '1.0a', '1.0b', and '1.0rc' are valid PEP 440 versions that should be treated as '1.0a0', '1.0b0', and '1.0rc0' respectively. However, the current implementation crashes when encountering these valid formats.

The bug occurs because the regex split pattern `r'(\D+)'` on line 604 produces an empty string at the end when a pre-release tag appears without a following number. When the code reaches line 611, it attempts to convert this empty string to an integer, causing the ValueError.

## Relevant Context

The function is located in `/Cython/Utils.py` at line 594-621. The crash occurs specifically at line 611 where `int(segment)` is called on an empty string.

The regex pattern `r'(\D+)'` splits on non-digit sequences and includes them in the result. For input '0.0rc', this produces: `['0', '.', '0', 'rc', '']`. The final empty string comes from the split operation when there are no digits after 'rc'.

Testing shows that versions with explicit numbers work correctly:
- '1.0rc1' → 0x010000C1 ✓
- '1.0a2' → 0x010000A2 ✓
- '1.0b3' → 0x010000B3 ✓

But versions without explicit numbers fail:
- '0.0rc' → ValueError ✗
- '0.0a' → ValueError ✗
- '1.0rc' → ValueError ✗
- '2.3.4a' → ValueError ✗

PEP 440 documentation: https://peps.python.org/pep-0440/#pre-releases

## Proposed Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -607,7 +607,7 @@ def build_hex_version(version_string):
             digits = (digits + [0, 0])[:3]  # 1.2a1 -> 1.2.0a1
         elif segment in ('.dev', '.pre', '.post'):
             break  # break since those are the last segments
-        elif segment != '.':
+        elif segment and segment != '.':
             digits.append(int(segment))

     digits = (digits + [0] * 3)[:4]
```