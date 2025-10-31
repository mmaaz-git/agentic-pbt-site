# Bug Report: dask.utils.natural_sort_key Unicode Digit Crash

**Target**: `dask.utils.natural_sort_key`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `natural_sort_key` function crashes with a `ValueError` when processing strings containing Unicode digit characters like '²', '³', or '①' because it uses `str.isdigit()` to detect digits but then calls `int()` which only accepts ASCII digits.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import natural_sort_key

@given(st.text())
def test_natural_sort_key_deterministic(s):
    result1 = natural_sort_key(s)
    result2 = natural_sort_key(s)
    assert result1 == result2
```

<details>

<summary>
**Failing input**: `'²'`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 17, in <module>
    test_natural_sort_key_deterministic()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "<string>", line 6, in test_natural_sort_key_deterministic
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "<string>", line 9, in test_natural_sort_key_deterministic
    result1 = natural_sort_key(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/dask/utils.py", line 1582, in natural_sort_key
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s)]
            ~~~^^^^^^
ValueError: invalid literal for int() with base 10: '²'
Falsifying example: test_natural_sort_key_deterministic(
    s='²',
)
Found failure with input: '²'
Error: invalid literal for int() with base 10: '²'
Found failure with input: '²'
Error: invalid literal for int() with base 10: '²'
Found failure with input: '²'
Error: invalid literal for int() with base 10: '²'
```
</details>

## Reproducing the Bug

```python
from dask.utils import natural_sort_key

# Test with superscript 2
print("Testing natural_sort_key('²')...")
try:
    result = natural_sort_key('²')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting natural_sort_key('file²name')...")
try:
    result = natural_sort_key('file²name')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting sorted(['file²', 'file1'], key=natural_sort_key)...")
try:
    result = sorted(['file²', 'file1'], key=natural_sort_key)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with other Unicode digits
print("\nTesting with circled digit '①'...")
try:
    result = natural_sort_key('①')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting with cube '³'...")
try:
    result = natural_sort_key('³')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError crashes on Unicode digit characters
</summary>
```
Testing natural_sort_key('²')...
Error: ValueError: invalid literal for int() with base 10: '²'

Testing natural_sort_key('file²name')...
Result: ['file²name']

Testing sorted(['file²', 'file1'], key=natural_sort_key)...
Result: ['file1', 'file²']

Testing with circled digit '①'...
Error: ValueError: invalid literal for int() with base 10: '①'

Testing with cube '³'...
Error: ValueError: invalid literal for int() with base 10: '³'
```
</details>

## Why This Is A Bug

The function violates its documented contract and reasonable user expectations in several ways:

1. **Function signature accepts any string**: The function is typed as `natural_sort_key(s: str) -> list[str | int]`, indicating it should handle any valid Python string without raising exceptions.

2. **No documented Unicode restrictions**: Neither the function signature nor the docstring mentions that Unicode digit characters are unsupported or will cause crashes.

3. **Inconsistent digit detection**: The bug stems from a mismatch between Python's `str.isdigit()` method and `int()` constructor:
   - `str.isdigit()` returns `True` for Unicode digits like '²', '³', '①', '⑨'
   - `int()` only accepts ASCII digits '0'-'9' and raises `ValueError` for Unicode digits

4. **Breaks natural sorting expectation**: The function is meant to provide natural sort ordering for strings, which is particularly important for internationalized applications that may encounter various Unicode characters in filenames or data.

5. **Partial failure mode**: Interestingly, strings like 'file²name' work correctly because the regex `r"(\d+)"` only matches ASCII digits, so '²' doesn't get split out as a separate part. The crash only occurs when Unicode digits appear as standalone parts after splitting.

## Relevant Context

The natural_sort_key function is located in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` at line 1582.

The problematic code uses a list comprehension that attempts to convert any part that passes `isdigit()` to an integer:
```python
return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s)]
```

Python's string methods have subtle differences:
- `str.isdigit()`: Returns True for Unicode digit characters (category Nd)
- `str.isdecimal()`: Returns True only for decimal numbers that can form base-10 numbers
- `str.isnumeric()`: Returns True for numeric characters including fractions

The `int()` constructor only accepts characters that `isdecimal()` returns True for, making it the appropriate check for this use case.

## Proposed Fix

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1579,7 +1579,7 @@ def natural_sort_key(s: str) -> list[str | int]:
     >>> sorted(a, key=natural_sort_key)
     ['f0', 'f1', 'f2', 'f8', 'f9', 'f10', 'f11', 'f19', 'f20', 'f21']
     """
-    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s)]
+    return [int(part) if part.isdecimal() else part for part in re.split(r"(\d+)", s)]


 def parse_bytes(s: float | str) -> int:
```