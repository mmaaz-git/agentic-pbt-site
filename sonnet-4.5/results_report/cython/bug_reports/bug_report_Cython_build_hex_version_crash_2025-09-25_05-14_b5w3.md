# Bug Report: Cython.Utils.build_hex_version - Crash on Missing Number After Release Tag

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`build_hex_version` crashes with `ValueError` when a version string has a release tag (a, b, rc) without a following number.

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
```

**Failing input**: `'0.0rc'`

## Reproducing the Bug

```python
from Cython.Utils import build_hex_version

version = '0.0rc'
result = build_hex_version(version)
```

**Output:**
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "Cython/Utils.py", line 611, in build_hex_version
    digits.append(int(segment))
ValueError: invalid literal for int() with base 10: ''
```

## Why This Is A Bug

According to PEP 440 (referenced in the docstring), version strings like "1.0rc1" or "1.0a2" are valid, but so is "1.0rc" without a number (equivalent to "1.0rc0"). The parsing logic on lines 604-611 splits on `\D+` (non-digits) which produces empty strings when release tags appear at the end without numbers. Line 611 tries to convert this empty string to an int, causing the crash.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -604,12 +604,13 @@ def build_hex_version(version_string):
     for segment in re.split(r'(\D+)', version_string):
         if segment in ('a', 'b', 'rc'):
             release_status = {'a': 0xA0, 'b': 0xB0, 'rc': 0xC0}[segment]
             digits = (digits + [0, 0])[:3]  # 1.2a1 -> 1.2.0a1
         elif segment in ('.dev', '.pre', '.post'):
             break  # break since those are the last segments
-        elif segment != '.':
+        elif segment and segment != '.':
             digits.append(int(segment))

     digits = (digits + [0] * 3)[:4]
     digits[3] += release_status
```