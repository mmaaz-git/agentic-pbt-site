# Bug Report: numpy.strings.replace Silently Truncates Replacement Strings

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.strings.replace` function silently truncates replacement strings when they exceed the input array's dtype width, causing data loss without warning or error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import numpy.strings as nps

@given(st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x),
       st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x),
       st.text(max_size=10).filter(lambda x: '\x00' not in x))
@settings(max_examples=500)
def test_replace_matches_python(prefix, old, suffix):
    assume(len(old) > 0)
    assume(len(prefix) + len(old) + len(suffix) < 50)

    s = prefix + old + suffix
    new = 'X' * (len(old) + 5) if len(old) < 10 else 'Y'

    arr = np.array([s])
    result = nps.replace(arr, old, new, count=1)
    python_result = s.replace(old, new, 1)

    assert result[0] == python_result, f"NumPy: {result[0]!r}, Python: {python_result!r}"

# Run the test
test_replace_matches_python()
```

<details>

<summary>
**Failing input**: `prefix='0', old='0', suffix=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 23, in <module>
    test_replace_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 6, in test_replace_matches_python
    st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 20, in test_replace_matches_python
    assert result[0] == python_result, f"NumPy: {result[0]!r}, Python: {python_result!r}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: NumPy: np.str_('XX0'), Python: 'XXXXXX0'
Falsifying example: test_replace_matches_python(
    prefix='0',
    old='0',
    suffix='',
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

s = '00'
arr = np.array([s])
result = nps.replace(arr, '0', 'XXXXXX', count=1)

print(f"Input: {s!r}")
print(f"Python result: {s.replace('0', 'XXXXXX', 1)!r}")
print(f"NumPy result:  {result[0]!r}")

assert result[0] == 'XXXXXX0', f"Expected 'XXXXXX0', got {result[0]!r}"
```

<details>

<summary>
AssertionError: Expected 'XXXXXX0', got np.str_('XX0')
</summary>
```
Input: '00'
Python result: 'XXXXXX0'
NumPy result:  np.str_('XX0')
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/repo.py", line 12, in <module>
    assert result[0] == 'XXXXXX0', f"Expected 'XXXXXX0', got {result[0]!r}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'XXXXXX0', got np.str_('XX0')
```
</details>

## Why This Is A Bug

The numpy.strings.replace function documentation explicitly states it returns "a copy of the string with occurrences of substring old replaced by new" and references Python's str.replace in the "See Also" section, implying equivalent behavior. However, the function violates this contract by silently truncating replacement strings.

The bug occurs because the function prematurely casts the `new` parameter to the input array's dtype before calculating the required buffer size. When the input array `['00']` has dtype `<U2` (2-character Unicode), the replacement string `'XXXXXX'` gets truncated to `'XX'` before buffer calculation. This leads to:

1. Incorrect buffer size calculation: `2 + 1 * (2 - 1) = 3` instead of `2 + 1 * (6 - 1) = 7`
2. Output array created with insufficient dtype `<U3` instead of `<U7`
3. Silent truncation of the correct result from `'XXXXXX0'` to `'XX0'`

This violates the principle of least surprise and causes silent data loss without any warning or error message.

## Relevant Context

The issue is in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py:1358-1359`, where scalar string arguments are cast to the input array's dtype before buffer size calculation. The NumPy documentation does not mention this truncation behavior anywhere, and the examples provided don't demonstrate cases where truncation would occur.

Python's str.replace, which this function claims to emulate element-wise, correctly handles replacement strings of any length without truncation. Users reasonably expect NumPy's version to behave similarly.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.strings.replace.html

## Proposed Fix

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1355,13 +1355,14 @@ def replace(a, old, new, count=-1):
         return _replace(arr, old, new, count)

     a_dt = arr.dtype
-    old = old.astype(old_dtype or a_dt, copy=False)
-    new = new.astype(new_dtype or a_dt, copy=False)
     max_int64 = np.iinfo(np.int64).max
     counts = _count_ufunc(arr, old, 0, max_int64)
     counts = np.where(count < 0, counts, np.minimum(counts, count))
     buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
     out_dtype = f"{arr.dtype.char}{buffersizes.max()}"
     out = np.empty_like(arr, shape=buffersizes.shape, dtype=out_dtype)
+
+    old = old.astype(old_dtype or a_dt, copy=False)
+    new = new.astype(new_dtype or a_dt, copy=False)

     return _replace(arr, old, new, counts, out=out)
```