# Bug Report: Cython.Utils.build_hex_version Version Collision with High Serial Numbers

**Target**: `Cython.Utils.build_hex_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `build_hex_version` function incorrectly adds release status bytes and serial numbers instead of using bitwise operations, causing version collisions where pre-release versions with high serial numbers produce the same hex value as final releases.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

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

    assert val_final > val_alpha, f"Final version {version_final} ({hex_final}) should be greater than alpha {version_alpha} ({hex_alpha})"

if __name__ == "__main__":
    test_release_status_ordering_alpha_vs_final()
```

<details>

<summary>
**Failing input**: `major=0, minor=0, alpha_ver=80`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 23, in <module>
    test_release_status_ordering_alpha_vs_final()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 8, in test_release_status_ordering_alpha_vs_final
    st.integers(min_value=0, max_value=99),
            ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 20, in test_release_status_ordering_alpha_vs_final
    assert val_final > val_alpha, f"Final version {version_final} ({hex_final}) should be greater than alpha {version_alpha} ({hex_alpha})"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: Final version 0.0 (0x000000F0) should be greater than alpha 0.0a80 (0x000000F0)
Falsifying example: test_release_status_ordering_alpha_vs_final(
    major=0,
    minor=0,
    alpha_ver=80,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import build_hex_version

# Test version collisions with high serial numbers
alpha80 = build_hex_version("0.0a80")
final = build_hex_version("0.0")
print(f"0.0a80: {alpha80}")
print(f"0.0:    {final}")
print(f"Equal: {alpha80 == final}")
print()

beta64 = build_hex_version("0.0b64")
print(f"0.0b64: {beta64}")
print(f"Equal to final: {beta64 == final}")
print()

rc48 = build_hex_version("0.0rc48")
print(f"0.0rc48: {rc48}")
print(f"Equal to final: {rc48 == final}")
print()

# Show the hex values as integers for comparison
print("Integer values:")
print(f"0.0a80: {int(alpha80, 16)}")
print(f"0.0:    {int(final, 16)}")
print(f"0.0b64: {int(beta64, 16)}")
print(f"0.0rc48: {int(rc48, 16)}")
```

<details>

<summary>
Version collision: multiple pre-release versions equal final version
</summary>
```
0.0a80: 0x000000F0
0.0:    0x000000F0
Equal: True

0.0b64: 0x000000F0
Equal to final: True

0.0rc48: 0x000000F0
Equal to final: True

Integer values:
0.0a80: 240
0.0:    240
0.0b64: 240
0.0rc48: 240
```
</details>

## Why This Is A Bug

The function violates the fundamental principle of version ordering where pre-release versions (alpha, beta, rc) should always be less than final releases. The bug occurs because the function incorrectly adds the release status value to the serial number instead of using proper bitwise encoding.

The function's docstring at line 596 states it works "like PY_VERSION_HEX", but the implementation differs significantly from Python's standard. In Python's PY_VERSION_HEX format:
- The last byte encodes: `(release_level << 4) | serial`
- Release levels: 0xA (alpha), 0xB (beta), 0xC (rc), 0xF (final)
- Serial numbers fit in 4 bits (0-15)

However, Cython's implementation at line 614 uses addition: `digits[3] += release_status`. This causes:
- `0.0a80`: 80 + 0xA0 (160) = 240 (0xF0) - collides with final!
- `0.0b64`: 64 + 0xB0 (176) = 240 (0xF0) - collides with final!
- `0.0rc48`: 48 + 0xC0 (192) = 240 (0xF0) - collides with final!
- Final `0.0`: 0 + 0xF0 (240) = 240 (0xF0)

This breaks version comparison logic, potentially affecting dependency resolution, version constraints, and any system that relies on monotonic version ordering.

## Relevant Context

The bug is found in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utils.py` at line 614. The function processes version strings according to PEP 440 but encodes them incorrectly.

Key observations:
- The release status values are defined at line 606: `{'a': 0xA0, 'b': 0xB0, 'rc': 0xC0}`
- Final releases get 0xF0 (line 603)
- Serial numbers above certain thresholds cause collisions:
  - Alpha: serial >= 80 collides (80 + 160 = 240)
  - Beta: serial >= 64 collides (64 + 176 = 240)
  - RC: serial >= 48 collides (48 + 192 = 240)
- Serial numbers > 255 - release_status will cause byte overflow into adjacent bytes

PEP 440 reference: https://peps.python.org/pep-0440/#public-version-identifiers

## Proposed Fix

```diff
--- a/Cython/Utils.py
+++ b/Cython/Utils.py
@@ -611,7 +611,8 @@ def build_hex_version(version_string):
             digits.append(int(segment))

     digits = (digits + [0] * 3)[:4]
-    digits[3] += release_status
+    # Use bitwise OR to properly encode release status with serial, matching PY_VERSION_HEX format
+    digits[3] = release_status | (digits[3] & 0x0F)  # Clamp serial to 4 bits

     # Then, build a single hex value, two hex digits per version part.
     hexversion = 0
```