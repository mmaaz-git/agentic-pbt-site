# Bug Report: dask.utils.parse_bytes Accepts Invalid Whitespace-Only Input

**Target**: `dask.utils.parse_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `parse_bytes` function silently accepts empty strings and whitespace-only strings, returning `1` instead of raising a `ValueError` as it does for other invalid inputs, violating the expected input validation contract.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for dask.utils.parse_bytes whitespace handling."""

from hypothesis import given, strategies as st, assume, settings, HealthCheck
import dask.utils
import pytest


@given(st.text(st.characters(whitelist_categories=('Zs', 'Cc')), min_size=1))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_parse_bytes_rejects_whitespace_only(s):
    assume(s.strip() == '')

    with pytest.raises(ValueError):
        dask.utils.parse_bytes(s)


if __name__ == "__main__":
    test_parse_bytes_rejects_whitespace_only()
```

<details>

<summary>
**Failing input**: `' '`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 19, in <module>
    test_parse_bytes_rejects_whitespace_only()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 10, in test_parse_bytes_rejects_whitespace_only
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 14, in test_parse_bytes_rejects_whitespace_only
    with pytest.raises(ValueError):
         ~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'ValueError'>
Falsifying example: test_parse_bytes_rejects_whitespace_only(
    s=' ',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py:1633
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of dask.utils.parse_bytes bug with whitespace-only strings."""

import dask.utils

# Test empty string
print("Testing empty string '':")
try:
    result = dask.utils.parse_bytes('')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test carriage return
print("\nTesting carriage return '\\r':")
try:
    result = dask.utils.parse_bytes('\r')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test newline
print("\nTesting newline '\\n':")
try:
    result = dask.utils.parse_bytes('\n')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test tab
print("\nTesting tab '\\t':")
try:
    result = dask.utils.parse_bytes('\t')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test space
print("\nTesting space ' ':")
try:
    result = dask.utils.parse_bytes(' ')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test multiple spaces
print("\nTesting multiple spaces '   ':")
try:
    result = dask.utils.parse_bytes('   ')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Control: Test invalid unit (should raise ValueError)
print("\nControl test with invalid unit '5 foos':")
try:
    result = dask.utils.parse_bytes('5 foos')
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError: {e}")
```

<details>

<summary>
All whitespace inputs incorrectly return 1 instead of raising ValueError
</summary>
```
Testing empty string '':
  Result: 1

Testing carriage return '\r':
  Result: 1

Testing newline '\n':
  Result: 1

Testing tab '\t':
  Result: 1

Testing space ' ':
  Result: 1

Testing multiple spaces '   ':
  Result: 1

Control test with invalid unit '5 foos':
  ValueError: Could not interpret 'foos' as a byte unit
```
</details>

## Why This Is A Bug

This violates the expected behavior and input validation contract of `parse_bytes`:

1. **Inconsistent error handling**: The function's docstring demonstrates that invalid inputs like `'5 foos'` raise `ValueError: Could not interpret 'foos' as a byte unit`. Whitespace-only and empty strings are similarly invalid (they don't represent any meaningful byte value) but are silently accepted.

2. **Principle of least surprise violation**: Users would not expect an empty string or whitespace characters to mean "1 byte". This behavior could mask real errors where empty/whitespace strings are passed accidentally.

3. **Unintended code interaction**: The bug occurs due to an interaction between three implementation details:
   - Line 1616: `s.replace(" ", "")` removes spaces but preserves other whitespace
   - Lines 1617-1618: When no digits are found, `"1"` is prepended to the string
   - Line 1629: Python's `float()` function silently strips trailing whitespace characters
   - Line 1654: The `byte_sizes` dict maps empty string `''` to `1`

   For example, `parse_bytes('\r')` becomes `'1\r'` after prepending, then `float('1\r')` returns `1.0`, and with empty suffix `''`, `byte_sizes['']` returns `1`.

4. **API contract violation**: A function that parses "byte strings" should validate that the input actually represents a byte value. Empty and whitespace-only strings do not meet this criterion.

## Relevant Context

The function is located at `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py:1585-1639`.

The `byte_sizes` dictionary (lines 1642-1658) intentionally maps empty string to 1, which is reasonable for supporting unit-only inputs like `'MB'` (meaning 1 MB). However, this creates the unintended side effect where pure whitespace also maps to 1 byte.

Python's `float()` function behavior with whitespace is documented but often surprising - it strips both leading and trailing whitespace, so `float('1\n')`, `float('1\r')`, `float('1\t')` all return `1.0`.

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1613,6 +1613,8 @@ def parse_bytes(s: float | str) -> int:
     """
     if isinstance(s, (int, float)):
         return int(s)
+    if not s or not s.strip():
+        raise ValueError(f"Could not interpret {s!r} as a byte string")
     s = s.replace(" ", "")
     if not any(char.isdigit() for char in s):
         s = "1" + s
```