# Bug Report: dask.utils.parse_timedelta Crashes with IndexError on Empty String Input

**Target**: `dask.utils.parse_timedelta`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_timedelta` function crashes with an `IndexError: string index out of range` when given an empty string or a string containing only spaces, instead of raising an informative `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import dask.utils
import pytest


@given(st.just(''))
@settings(max_examples=1)
def test_parse_timedelta_empty_string(s):
    """Test that parse_timedelta raises ValueError on empty string input."""
    with pytest.raises(ValueError):
        dask.utils.parse_timedelta(s)

if __name__ == "__main__":
    # Run the test
    test_parse_timedelta_empty_string()
```

<details>

<summary>
**Failing input**: `''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 15, in <module>
    test_parse_timedelta_empty_string()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 7, in test_parse_timedelta_empty_string
    @settings(max_examples=1)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 11, in test_parse_timedelta_empty_string
    dask.utils.parse_timedelta(s)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1869, in parse_timedelta
    if not s[0].isdigit():
           ~^^^
IndexError: string index out of range
Falsifying example: test_parse_timedelta_empty_string(
    s='',
)
```
</details>

## Reproducing the Bug

```python
import dask.utils

# Test with empty string
try:
    result = dask.utils.parse_timedelta('')
    print(f"Empty string result: {result}")
except Exception as e:
    print(f"Empty string error: {type(e).__name__}: {e}")

# Test with space-only string
try:
    result = dask.utils.parse_timedelta(' ')
    print(f"Space-only string result: {result}")
except Exception as e:
    print(f"Space-only string error: {type(e).__name__}: {e}")

# Test with multiple spaces
try:
    result = dask.utils.parse_timedelta('   ')
    print(f"Multiple spaces result: {result}")
except Exception as e:
    print(f"Multiple spaces error: {type(e).__name__}: {e}")
```

<details>

<summary>
IndexError crash on empty/space-only strings
</summary>
```
Empty string error: IndexError: string index out of range
Space-only string error: IndexError: string index out of range
Multiple spaces error: IndexError: string index out of range
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Uninformative error**: The function crashes with `IndexError` instead of raising a descriptive `ValueError` that would help users understand what went wrong.

2. **Inconsistent error handling**: The function already raises `ValueError` for other invalid inputs (e.g., when `default=False` and no unit is provided), and `KeyError` with detailed messages for invalid time units. Empty strings should follow this pattern.

3. **Poor defensive programming**: The code accesses `s[0]` without checking if the string is empty after removing spaces, which is a classic bounds-checking error.

4. **Documentation expectations**: While the documentation doesn't explicitly state how empty strings should be handled, crashing with `IndexError` is clearly unintended behavior for a public API function.

5. **Comparison with similar functions**: Other parsing functions in the same module handle edge cases more gracefully, without crashing on empty input.

## Relevant Context

The crash occurs at line 1869 in `/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py`:

```python
s = s.replace(" ", "")  # Line 1868: removes all spaces
if not s[0].isdigit():  # Line 1869: crashes here if s is empty
    s = "1" + s
```

After removing spaces from the input string, the code attempts to access the first character without verifying the string is non-empty. This is particularly problematic for strings that contain only spaces, as they become empty after the `replace()` operation.

The function documentation shows examples like `'3s'`, `'3.5 seconds'`, and `'300ms'`, suggesting it expects non-empty strings with time values. However, proper error handling for invalid input is still expected.

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1866,6 +1866,9 @@ def parse_timedelta(s, default="seconds"):
     if isinstance(s, Number):
         s = str(s)
     s = s.replace(" ", "")
+
+    if not s:
+        raise ValueError(f"Could not interpret empty string as a time delta")
+
     if not s[0].isdigit():
         s = "1" + s
```