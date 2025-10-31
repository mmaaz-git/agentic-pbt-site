# Bug Report: Cython.Utils.build_hex_version Overflow

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`build_hex_version` silently produces incorrect hex version identifiers when version components exceed 255 or when prerelease serial numbers cause byte overflow. The function generates multi-byte values that overflow into adjacent version components, violating the PY_VERSION_HEX format.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example

@given(st.integers(min_value=0, max_value=100000))
@settings(max_examples=500)
@example(256)
@example(1000)
def test_large_version_numbers(major):
    version_str = str(major)
    result = build_hex_version(version_str)
    value = int(result, 16)
    assert 0 <= value <= 0xFFFFFFFF

@given(
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.sampled_from(['a', 'b', 'rc']),
    st.integers(min_value=0, max_value=255)
)
@settings(max_examples=500)
def test_prerelease_versions(major, minor, release_type, serial):
    version_str = f"{major}.{minor}{release_type}{serial}"
    result = build_hex_version(version_str)
    value = int(result, 16)
    extracted_micro = (value >> 8) & 0xFF
    assert extracted_micro == 0
```

**Failing inputs**:
- Bug 1: `major=256` produces `0x1000000F0` (exceeds 32-bit)
- Bug 2: `"0.0a96"` produces `0x00000100` (micro=1 instead of 0)

## Reproducing the Bug

```python
from Cython.Utils import build_hex_version

print("Bug 1: Large version numbers")
result = build_hex_version("256")
print(f"  Input: '256'")
print(f"  Output: {result}")
print(f"  Value: {int(result, 16)} (exceeds 0xFFFFFFFF)")

result = build_hex_version("1000")
print(f"  Input: '1000'")
print(f"  Output: {result}")
print(f"  Value: {int(result, 16)} (exceeds 0xFFFFFFFF)")

print("\nBug 2: Prerelease serial overflow")
result = build_hex_version("0.0a96")
value = int(result, 16)
print(f"  Input: '0.0a96'")
print(f"  Output: {result}")
print(f"  Extracted micro: {(value >> 8) & 0xFF} (expected 0)")
print(f"  Explanation: 96 + 0xA0 = 256, overflows byte")

result = build_hex_version("0.0a160")
value = int(result, 16)
print(f"  Input: '0.0a160'")
print(f"  Output: {result}")
print(f"  Extracted micro: {(value >> 8) & 0xFF} (expected 0)")
```

## Why This Is A Bug

The function is designed to produce PY_VERSION_HEX-style identifiers where each version component occupies one byte (0-255). The bugs violate this format:

**Bug 1**: Version components > 255 overflow their byte boundaries, producing values that exceed the 32-bit format and corrupt adjacent components.

**Bug 2**: The serial number addition on line 614 (`digits[3] += release_status`) can overflow when `serial + release_status > 255`. For alpha releases (0xA0 = 160), any serial >= 96 overflows. The overflow bits spill into the micro version field.

The function neither validates inputs nor handles overflow, producing silently corrupted output.

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -594,6 +594,9 @@ def build_hex_version(version_string):
     """
     Parse and translate public version identifier like '4.3a1' into the readable hex representation '0x040300A1' (like PY_VERSION_HEX).

+    Note: Version components (major, minor, micro) must be in range 0-255.
+    Prerelease serial numbers must be in range 0-95 (alpha), 0-79 (beta), or 0-63 (rc).
+
     SEE: https://peps.python.org/pep-0440/#public-version-identifiers
     """
     # Parse '4.12a1' into [4, 12, 0, 0xA01]
@@ -608,10 +611,17 @@ def build_hex_version(version_string):
         elif segment in ('.dev', '.pre', '.post'):
             break  # break since those are the last segments
         elif segment != '.':
-            digits.append(int(segment))
+            num = int(segment)
+            if num > 255:
+                raise ValueError(f"Version component {num} exceeds maximum 255 in '{version_string}'")
+            digits.append(num)

     digits = (digits + [0] * 3)[:4]
-    digits[3] += release_status
+    serial_with_status = digits[3] + release_status
+    if serial_with_status > 255:
+        max_serial = 255 - release_status
+        raise ValueError(f"Prerelease serial {digits[3]} exceeds maximum {max_serial} for release type in '{version_string}'")
+    digits[3] = serial_with_status

     # Then, build a single hex value, two hex digits per version part.
     hexversion = 0
```