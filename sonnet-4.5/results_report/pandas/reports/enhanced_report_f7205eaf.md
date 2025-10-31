# Bug Report: pandas.core.strings.slice_replace Incorrect Behavior When start > stop

**Target**: `pandas.core.strings.accessor.StringMethods.slice_replace`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `slice_replace` method in pandas produces incorrect results when `start > stop`, deviating from Python's standard slicing semantics by incorrectly adjusting the stop position instead of treating the slice as empty.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis property-based test for pandas slice_replace bug
"""
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.text(max_size=10)
)
@settings(max_examples=500)
def test_slice_replace_matches_python(strings, start, stop, repl):
    s = pd.Series(strings)
    pandas_result = s.str.slice_replace(start, stop, repl)

    for i in range(len(s)):
        if isinstance(s.iloc[i], str):
            original = s.iloc[i]
            expected = original[:start] + repl + original[stop:]
            assert pandas_result.iloc[i] == expected


if __name__ == "__main__":
    # Run the test
    test_slice_replace_matches_python()
```

<details>

<summary>
**Failing input**: `strings=['0'], start=1, stop=0, repl=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 29, in <module>
    test_slice_replace_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 10, in test_slice_replace_matches_python
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 24, in test_slice_replace_matches_python
    assert pandas_result.iloc[i] == expected
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_slice_replace_matches_python(
    strings=['0'],
    start=1,
    stop=0,
    repl='',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the pandas slice_replace bug
"""
import pandas as pd

print("Testing pandas.core.strings.slice_replace bug")
print("=" * 60)

# Test case 1: Simple case with start > stop
print("\nTest case 1: strings=['0'], start=1, stop=0, repl=''")
s = pd.Series(['0'])
result = s.str.slice_replace(start=1, stop=0, repl='')

print(f"Pandas result:   {result.iloc[0]!r}")
print(f"Expected result: {'0'[:1] + '' + '0'[0:]!r}")
print(f"Are they equal?  {result.iloc[0] == ('0'[:1] + '' + '0'[0:])}")

# Verify Python slicing behavior
print(f"\nPython slicing verification:")
print(f"'0'[1:0] = {'0'[1:0]!r} (empty string)")
print(f"'0'[:1] = {'0'[:1]!r}")
print(f"'0'[0:] = {'0'[0:]!r}")
print(f"'0'[:1] + '' + '0'[0:] = {'0'[:1] + '' + '0'[0:]!r}")

print("\n" + "-" * 60)

# Test case 2: More complex case with replacement string
print("\nTest case 2: strings=['hello'], start=3, stop=1, repl='X'")
s2 = pd.Series(['hello'])
result2 = s2.str.slice_replace(start=3, stop=1, repl='X')

print(f"Pandas result:   {result2.iloc[0]!r}")
print(f"Expected result: {'hello'[:3] + 'X' + 'hello'[1:]!r}")
print(f"Are they equal?  {result2.iloc[0] == ('hello'[:3] + 'X' + 'hello'[1:])}")

# Verify Python slicing behavior
print(f"\nPython slicing verification:")
print(f"'hello'[3:1] = {'hello'[3:1]!r} (empty string)")
print(f"'hello'[:3] = {'hello'[:3]!r}")
print(f"'hello'[1:] = {'hello'[1:]!r}")
print(f"'hello'[:3] + 'X' + 'hello'[1:] = {'hello'[:3] + 'X' + 'hello'[1:]!r}")

print("\n" + "=" * 60)
print("Summary: The bug occurs when start > stop.")
print("Pandas incorrectly adjusts the stop position, producing wrong results.")
```

<details>

<summary>
AssertionError: Pandas result does not match expected Python slicing behavior
</summary>
```
Testing pandas.core.strings.slice_replace bug
============================================================

Test case 1: strings=['0'], start=1, stop=0, repl=''
Pandas result:   '0'
Expected result: '00'
Are they equal?  False

Python slicing verification:
'0'[1:0] = '' (empty string)
'0'[:1] = '0'
'0'[0:] = '0'
'0'[:1] + '' + '0'[0:] = '00'

------------------------------------------------------------

Test case 2: strings=['hello'], start=3, stop=1, repl='X'
Pandas result:   'helXlo'
Expected result: 'helXello'
Are they equal?  False

Python slicing verification:
'hello'[3:1] = '' (empty string)
'hello'[:3] = 'hel'
'hello'[1:] = 'ello'
'hello'[:3] + 'X' + 'hello'[1:] = 'helXello'

============================================================
Summary: The bug occurs when start > stop.
Pandas incorrectly adjusts the stop position, producing wrong results.
```
</details>

## Why This Is A Bug

The pandas documentation states that `slice_replace` "replaces a positional slice of a string with another value" and should be equivalent to `string[:start] + repl + string[stop:]`. However, the implementation contains special logic that modifies the stop position when `start > stop`, causing incorrect behavior.

In Python's standard slicing semantics:
- When `start >= stop`, the slice `[start:stop]` returns an empty string
- The operation `string[:start] + repl + string[stop:]` is well-defined and deterministic for all valid start/stop values
- Empty slices are a normal and expected part of Python's slicing behavior, not an error condition

The bug manifests because the implementation checks if `x[start:stop] == ""` and then sets `local_stop = start`, which incorrectly modifies the tail portion of the string that gets appended. This breaks the documented contract of the function and violates the principle of least surprise for Python developers.

## Relevant Context

The bug is located in `/pandas/core/strings/object_array.py` at lines 352-355 in the `_str_slice_replace` method. The problematic code checks if the slice is empty and adjusts the stop position:

```python
if x[start:stop] == "":
    local_stop = start
else:
    local_stop = stop
```

This logic is flawed because:
1. An empty slice can occur legitimately when `start >= stop`
2. The adjustment causes `x[local_stop:]` to return a different substring than `x[stop:]`
3. This violates the documented behavior that the operation should be equivalent to `string[:start] + repl + string[stop:]`

Documentation link: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.slice_replace.html

The method is part of pandas' string accessor functionality, which provides vectorized string operations on Series objects containing string data.

## Proposed Fix

```diff
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -350,15 +350,11 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
             repl = ""

         def f(x):
-            if x[start:stop] == "":
-                local_stop = start
-            else:
-                local_stop = stop
             y = ""
             if start is not None:
                 y += x[:start]
             y += repl
             if stop is not None:
-                y += x[local_stop:]
+                y += x[stop:]
             return y

         return self._str_map(f)
```