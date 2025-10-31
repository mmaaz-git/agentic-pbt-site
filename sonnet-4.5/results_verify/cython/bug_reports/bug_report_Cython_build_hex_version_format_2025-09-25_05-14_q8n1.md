# Bug Report: Cython.Utils.build_hex_version - Wrong Output Length

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`build_hex_version` produces hex strings with 9 digits instead of 8 for certain version numbers, violating the documented format '0x%08X'.

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

**Failing input**: `'0.70000'`

## Reproducing the Bug

```python
from Cython.Utils import build_hex_version

version = '0.70000'
result = build_hex_version(version)
print(f"Input:  {version}")
print(f"Output: {result}")
print(f"Length: {len(result)} (expected 10: '0x' + 8 hex digits)")
```

**Output:**
```
Input:  0.70000
Output: 0x1117000F0
Length: 11 (expected 10: '0x' + 8 hex digits)
```

## Why This Is A Bug

The function's docstring states it produces "readable hex representation '0x040300A1' (like PY_VERSION_HEX)" and line 621 uses format string `'0x%08X'` which should produce exactly 8 hex digits. However, for version '0.70000', it produces 9 hex digits. The format string on line 621 is correct, but the value being formatted (`0x1117000F0` = 4,581,515,504 in decimal) exceeds the maximum value that fits in 8 hex digits (0xFFFFFFFF = 4,294,967,295).

The issue is that the parsing logic allows version components to exceed 255 (0xFF). For '0.70000', it parses this as [0, 70000] instead of treating large numbers as invalid or clamping them.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -610,7 +610,11 @@ def build_hex_version(version_string):
         elif segment != '.':
             digits.append(int(segment))

-    digits = (digits + [0] * 3)[:4]
+    digits = (digits + [0] * 3)[:4]
+
+    for i, digit in enumerate(digits):
+        if digit > 255:
+            digits[i] = 255
+
     digits[3] += release_status

     # Then, build a single hex value, two hex digits per version part.
```