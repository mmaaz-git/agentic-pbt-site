# Bug Report: Cython.Utils.build_hex_version - Version Component Overflow

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_hex_version` function does not validate that version components fit within a single byte (0-255), causing it to produce hex strings longer than the documented 8 hex digits when version numbers >= 256 are provided.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utils import build_hex_version

@given(st.text(alphabet='0123456789.', min_size=1, max_size=20))
def test_build_hex_version_format(version_str):
    version_str = version_str.lstrip('.')
    if not version_str or not any(c.isdigit() for c in version_str):
        return

    result = build_hex_version(version_str)
    assert len(result) == 10, f"Expected 10 chars, got {len(result)}"
```

**Failing input**: `version_str='300'`

## Reproducing the Bug

```python
from Cython.Utils import build_hex_version

result = build_hex_version('300')
print(f"build_hex_version('300') = '{result}'")
print(f"Length: {len(result)} (expected 10)")
print(f"Hex digits: {len(result) - 2} (expected 8)")
```

Output:
```
build_hex_version('300') = '0x12C0000F0'
Length: 11 (expected 10)
Hex digits: 9 (expected 8)
```

Additional examples:
```python
print(build_hex_version('255'))
print(build_hex_version('256'))
print(build_hex_version('1000'))
```

Output:
```
0xFF0000F0
0x1000000F0
0x3E80000F0
```

## Why This Is A Bug

The function's docstring states it produces "readable hex representation '0x040300A1' (like PY_VERSION_HEX)", which is always 10 characters (0x + 8 hex digits = 4 bytes). The comment on line 616 says "two hex digits per version part", meaning each component should fit in one byte (0-255).

However, the function does not validate this constraint. When a version component exceeds 255, it overflows into the next byte, producing output with more than 8 hex digits.

The format specifier `'0x%08X'` on line 621 means "at least 8 hex digits", so values requiring more digits will produce longer strings, violating the documented format.

## Fix

Add validation to ensure version components fit in a byte:

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -610,6 +610,10 @@ def build_hex_version(version_string):
         elif segment != '.':
             digits.append(int(segment))

+    for digit in digits:
+        if digit > 255:
+            raise ValueError(f"Version component {digit} exceeds maximum value 255")
+
     digits = (digits + [0] * 3)[:4]
     digits[3] += release_status
```

Alternatively, the function could silently truncate values to a byte:

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -610,6 +610,9 @@ def build_hex_version(version_string):
         elif segment != '.':
             digits.append(int(segment))

+    digits = [min(d, 255) for d in digits]
+
     digits = (digits + [0] * 3)[:4]
     digits[3] += release_status
```

The validation approach is preferable as it makes the constraint explicit rather than silently producing potentially incorrect output.