# Bug Report: dask.utils.format_bytes Violates 10-Character Output Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function produces 11-character output for values >= 1000 PiB, violating its documented guarantee that output is "always <= 10 characters" for values < 2^60.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for format_bytes output length invariant"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from hypothesis import given, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_output_length_invariant(n):
    """Test that format_bytes output is always <= 10 characters for values < 2^60"""
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

if __name__ == "__main__":
    # Run the test
    test_format_bytes_output_length_invariant()
```

<details>

<summary>
**Failing input**: `1125894277343089729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 18, in <module>
    test_format_bytes_output_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 11, in test_format_bytes_output_length_invariant
    def test_format_bytes_output_length_invariant(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 14, in test_format_bytes_output_length_invariant
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11 > 10
Falsifying example: test_format_bytes_output_length_invariant(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal demonstration of format_bytes bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import format_bytes

# The specific failing input from Hypothesis
n = 1125894277343089729

# Call the function
result = format_bytes(n)

# Display the results
print(f"Input value: {n}")
print(f"Is n < 2^60? {n < 2**60}")
print(f"Output: '{result}'")
print(f"Output length: {len(result)} characters")
print(f"Expected: <= 10 characters (per documentation)")
print()

# Verify the violation
assert n < 2**60, f"Value {n} is not less than 2^60"
assert len(result) == 11, f"Expected output length 11, got {len(result)}"
assert result == '1000.00 PiB', f"Expected '1000.00 PiB', got '{result}'"

print("Bug confirmed: Output exceeds documented 10-character limit")
```

<details>

<summary>
Output shows 11-character result exceeding documented limit
</summary>
```
Input value: 1125894277343089729
Is n < 2^60? True
Output: '1000.00 PiB'
Output length: 11 characters
Expected: <= 10 characters (per documentation)

Bug confirmed: Output exceeds documented 10-character limit

```
</details>

## Why This Is A Bug

The function's docstring explicitly guarantees: "For all values < 2**60, the output is always <= 10 characters." This is an unambiguous API contract that the implementation violates.

The bug occurs because the function formats all byte values with 2 decimal places using `f"{n / k:.2f} {prefix}B"`. When values reach 1000 or more PiB (petabytes), the output format changes from 3 digits before the decimal ("999.00 PiB" = 10 chars) to 4 digits ("1000.00 PiB" = 11 chars).

This violates the documented invariant for any value in the range [1000 * 2^50, 2^60), approximately 127 PiB worth of valid inputs. While these are extremely large values (over 1 exabyte), they are still within the explicitly documented valid input range where the 10-character guarantee should hold.

## Relevant Context

The format_bytes function is located in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` starting at line 1771.

The issue specifically affects the PiB (pebibyte) unit because:
1. It's the largest unit defined in the function (line 1791: `("Pi", 2**50)`)
2. There's no larger unit to convert to when values >= 1000 PiB
3. All smaller units (TiB, GiB, MiB, kiB) automatically convert to larger units before reaching 1000

The 10-character limit appears to be a deliberate design constraint, possibly for ensuring consistent formatting in tabular output or terminal displays where column alignment matters.

Documentation reference: https://docs.dask.org/en/latest/api.html#dask.utils.format_bytes

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1795,7 +1795,14 @@ def format_bytes(n: int) -> str:
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            value = n / k
+            # Dynamically adjust decimal places to maintain <= 10 characters
+            if value >= 1000:
+                return f"{value:.0f} {prefix}B"
+            elif value >= 100:
+                return f"{value:.1f} {prefix}B"
+            else:
+                return f"{value:.2f} {prefix}B"
     return f"{n} B"
```