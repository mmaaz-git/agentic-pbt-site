# Bug Report: Cython.Compiler.Naming.py_version_hex Integer Overflow

**Target**: `Cython.Compiler.Naming.py_version_hex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `py_version_hex` function silently corrupts data when version components >= 256 are passed, causing different inputs to produce identical outputs and making round-trip conversion impossible.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Compiler.Naming import py_version_hex

@given(
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=15),
    st.integers(min_value=0, max_value=15)
)
def test_py_version_hex_round_trip(major, minor, micro, level, serial):
    """Property: Version components should be extractable from hex value"""
    result = py_version_hex(major, minor, micro, level, serial)

    extracted_major = (result >> 24) & 0xFF
    extracted_minor = (result >> 16) & 0xFF
    extracted_micro = (result >> 8) & 0xFF
    extracted_level = (result >> 4) & 0xF
    extracted_serial = result & 0xF

    assert extracted_major == major
    assert extracted_minor == minor
    assert extracted_micro == micro
    assert extracted_level == level
    assert extracted_serial == serial
```

**Failing input**: `major=3, minor=256, micro=0` (and many others with components >= 256)

## Reproducing the Bug

```python
from Cython.Compiler.Naming import py_version_hex

v1 = py_version_hex(3, 0, 0)
v2 = py_version_hex(3, 256, 0)
print(f"py_version_hex(3, 0, 0)   = {hex(v1)}")
print(f"py_version_hex(3, 256, 0) = {hex(v2)}")
print(f"Same result? {v1 == v2}")

original_minor = 256
result = py_version_hex(3, original_minor, 0)
extracted_minor = (result >> 16) & 0xFF
print(f"Input: {original_minor}, Extracted: {extracted_minor}")
```

**Output**:
```
py_version_hex(3, 0, 0)   = 0x3000000
py_version_hex(3, 256, 0) = 0x3000000
Same result? True
Input: 256, Extracted: 0
```

## Why This Is A Bug

1. **Data Loss**: Different inputs `(3, 0, 0)` and `(3, 256, 0)` produce identical outputs
2. **Round-trip Failure**: `py_version_hex(3, 256, 0)` encodes but extracts back as `(3, 0, 0)`
3. **No Validation**: Function silently accepts out-of-range values without error or warning
4. **Silent Corruption**: Users won't know their data was corrupted

The function uses bit shifting without validating that components fit in their designated bit fields:
- Major: 8 bits (0-255)
- Minor: 8 bits (0-255)
- Micro: 8 bits (0-255)
- Level: 4 bits (0-15)
- Serial: 4 bits (0-15)

Values >= these limits overflow and wrap around.

## Fix

Add input validation to ensure version components are within valid ranges:

```diff
 def py_version_hex(major, minor=0, micro=0, release_level=0, release_serial=0):
+    if not (0 <= major <= 255):
+        raise ValueError(f"major version must be 0-255, got {major}")
+    if not (0 <= minor <= 255):
+        raise ValueError(f"minor version must be 0-255, got {minor}")
+    if not (0 <= micro <= 255):
+        raise ValueError(f"micro version must be 0-255, got {micro}")
+    if not (0 <= release_level <= 15):
+        raise ValueError(f"release_level must be 0-15, got {release_level}")
+    if not (0 <= release_serial <= 15):
+        raise ValueError(f"release_serial must be 0-15, got {release_serial}")
     return (major << 24) | (minor << 16) | (micro << 8) | (release_level << 4) | (release_serial)
```