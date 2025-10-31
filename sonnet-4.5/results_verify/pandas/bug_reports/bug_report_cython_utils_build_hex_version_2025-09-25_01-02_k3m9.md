# Bug Report: Cython.Utils.build_hex_version Version Collision and Overflow

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`build_hex_version` incorrectly adds release status bytes and serial numbers, causing version collisions and byte overflow that violate monotonicity and PY_VERSION_HEX compatibility.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utils import build_hex_version

@given(st.integers(min_value=0, max_value=99),
       st.integers(min_value=0, max_value=99),
       st.integers(min_value=1, max_value=99))
def test_release_status_ordering_alpha_vs_final(major, minor, alpha_ver):
    version_alpha = f"{major}.{minor}a{alpha_ver}"
    version_final = f"{major}.{minor}"

    hex_alpha = build_hex_version(version_alpha)
    hex_final = build_hex_version(version_final)

    val_alpha = int(hex_alpha, 16)
    val_final = int(hex_final, 16)

    assert val_final > val_alpha
```

**Failing inputs**:
- `major=0, minor=0, alpha_ver=80`: `0.0a80` == `0.0` (both produce 0x000000F0)
- `major=0, minor=0, beta_ver=64`: `0.0b64` == `0.0` (both produce 0x000000F0)
- `major=0, minor=0, rc_ver=48`: `0.0rc48` == `0.0` (both produce 0x000000F0)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import build_hex_version

alpha80 = build_hex_version("0.0a80")
final = build_hex_version("0.0")
print(f"0.0a80: {alpha80}")
print(f"0.0:    {final}")
print(f"Equal: {alpha80 == final}")

beta64 = build_hex_version("0.0b64")
print(f"0.0b64: {beta64}")
print(f"Equal to final: {beta64 == final}")

rc48 = build_hex_version("0.0rc48")
print(f"0.0rc48: {rc48}")
print(f"Equal to final: {rc48 == final}")
```

**Output**:
```
0.0a80: 0x000000F0
0.0:    0x000000F0
Equal: True
0.0b64: 0x000000F0
Equal to final: True
0.0rc48: 0x000000F0
Equal to final: True
```

## Why This Is A Bug

The function claims to follow PY_VERSION_HEX format (line 596) but implements it incorrectly.

**Root cause** (Utils.py:614):
```python
digits[3] += release_status
```

This ADDS the serial number and release status, causing:
- `0.0a80`: 80 + 0xA0 (160) = 240 = 0xF0 (collides with final!)
- `0.0b64`: 64 + 0xB0 (176) = 240 = 0xF0 (collides with final!)
- `0.0rc48`: 48 + 0xC0 (192) = 240 = 0xF0 (collides with final!)

**Python's PY_VERSION_HEX format**:
- Last byte = `(release_level << 4) | serial`
- Release level: 0xA (alpha), 0xB (beta), 0xC (rc), 0xF (final)
- Serial: 0-15 (fits in 4 bits)
- Example: `3.13rc2` → `(0xC << 4) | 2` = `0xC2`

**Violations**:
1. **Monotonicity**: Alpha/beta/rc versions with high serials can equal or exceed final versions
2. **PY_VERSION_HEX compatibility**: Format doesn't match Python's standard
3. **Byte overflow**: Serial numbers > 255 will overflow into adjacent bytes

## Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -610,8 +610,12 @@ def build_hex_version(version_string):
         elif segment != '.':
             digits.append(int(segment))

     digits = (digits + [0] * 3)[:4]
-    digits[3] += release_status
+    # Encode as (release_status << 4) | serial, matching PY_VERSION_HEX format
+    # Serial is clamped to 4 bits (0-15) to prevent overflow
+    serial = digits[3] & 0x0F  # Keep only lower 4 bits
+    digits[3] = release_status | serial

     # Then, build a single hex value, two hex digits per version part.
     hexversion = 0
```

**Alternative fix** (if full serial range is intended):
If Cython intentionally wants to support serial > 15 (unlike Python), the release status should use different encoding:
```diff
-    release_status = {'a': 0xA0, 'b': 0xB0, 'rc': 0xC0}[segment]
+    release_status = {'a': 0xA000, 'b': 0xB000, 'rc': 0xC000}[segment]
```
But this would be incompatible with PY_VERSION_HEX format.