# Bug Report: pandas.core.dtypes.dtypes.SparseDtype Asymmetric Equality Comparison

**Target**: `pandas.core.dtypes.dtypes.SparseDtype.__eq__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

SparseDtype's equality operator violates Python's fundamental requirement that equality be symmetric - when comparing SparseDtype objects with NaN vs non-NaN fill values, `dtype1 == dtype2` can return True while `dtype2 == dtype1` returns False.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from pandas.core.dtypes.dtypes import SparseDtype

@st.composite
def valid_sparse_dtypes(draw):
    base_dtype = draw(st.sampled_from([np.float32, np.float64]))
    use_default = draw(st.booleans())
    if use_default:
        return SparseDtype(base_dtype)
    fill_value = draw(st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.floats(allow_nan=False, allow_infinity=False)
    ))
    return SparseDtype(base_dtype, fill_value)

@given(valid_sparse_dtypes(), valid_sparse_dtypes())
def test_equality_symmetric(dtype1, dtype2):
    """Property: If dtype1 == dtype2, then dtype2 == dtype1"""
    if dtype1 == dtype2:
        assert dtype2 == dtype1, \
            f"Equality not symmetric: {dtype1} == {dtype2} but {dtype2} != {dtype1}"

if __name__ == "__main__":
    test_equality_symmetric()
```

<details>

<summary>
**Failing input**: `dtype1=Sparse[float32, nan]`, `dtype2=Sparse[float32, 0.0]`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1878: RuntimeWarning: overflow encountered in cast
  casted = dtype.type(element)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1879: RuntimeWarning: overflow encountered in cast
  if np.isnan(casted) or casted == element:
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1878: RuntimeWarning: overflow encountered in cast
  casted = dtype.type(element)
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/cast.py:1879: RuntimeWarning: overflow encountered in cast
  if np.isnan(casted) or casted == element:
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 25, in <module>
    test_equality_symmetric()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 18, in test_equality_symmetric
    def test_equality_symmetric(dtype1, dtype2):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 21, in test_equality_symmetric
    assert dtype2 == dtype1, \
           ^^^^^^^^^^^^^^^^
AssertionError: Equality not symmetric: Sparse[float32, nan] == Sparse[float32, 0.0] but Sparse[float32, 0.0] != Sparse[float32, nan]
Falsifying example: test_equality_symmetric(
    dtype1=Sparse[float32, nan],
    dtype2=Sparse[float32, 0.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/60/hypo.py:22
        /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/_dtype.py:345
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/base.py:113
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py:1716
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.dtypes.dtypes import SparseDtype

dtype1 = SparseDtype(np.float32, np.nan)
dtype2 = SparseDtype(np.float32, 0.0)

print(f"dtype1 = {dtype1}")
print(f"dtype2 = {dtype2}")
print()
print(f"dtype1 == dtype2: {dtype1 == dtype2}")
print(f"dtype2 == dtype1: {dtype2 == dtype1}")
print()
print("This demonstrates asymmetric equality - dtype1 == dtype2 returns True")
print("but dtype2 == dtype1 returns False, violating the symmetric property of equality.")
```

<details>

<summary>
Asymmetric equality comparison output
</summary>
```
dtype1 = Sparse[float32, nan]
dtype2 = Sparse[float32, 0.0]

dtype1 == dtype2: True
dtype2 == dtype1: False

This demonstrates asymmetric equality - dtype1 == dtype2 returns True
but dtype2 == dtype1 returns False, violating the symmetric property of equality.
```
</details>

## Why This Is A Bug

This violates Python's documented contract that equality must be symmetric. According to the Python data model documentation, if `a == b` returns True, then `b == a` must also return True. This asymmetry breaks several important invariants:

1. **Python's Equality Contract**: The Python documentation explicitly states that equality comparisons should be symmetric. This is a fundamental requirement for the `__eq__` method.

2. **Hash Consistency**: The code includes `_is_na_fill_value` in the metadata (line 1663 in dtypes.py) specifically to avoid hash collisions between `SparseDtype(float, 0.0)` and `SparseDtype(float, nan)`. However, the asymmetric equality means two objects that compare as equal might have different hashes, violating Python's requirement that equal objects must have equal hashes.

3. **Container Operations**: This bug can cause incorrect behavior when SparseDtype objects are used in sets, dictionaries, or other collections that rely on equality comparisons. For example, `dtype1 in {dtype2}` would return True, but `dtype2 in {dtype1}` would return False.

4. **Testing and Assertions**: Code using equality assertions like `assert dtype1 == dtype2` could pass while `assert dtype2 == dtype1` would fail, leading to confusing test failures and debugging challenges.

## Relevant Context

The root cause is in the `__eq__` method implementation (lines 1707-1717 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/dtypes/dtypes.py`):

- When `self._is_na_fill_value` is True (the dtype has a NaN fill value), the code uses a type-based check: it returns True if the other dtype either also has a NaN fill value OR if the fill values have compatible types (lines 1713-1717).

- When `self._is_na_fill_value` is False (the dtype has a non-NaN fill value), the code uses direct value equality: `self.fill_value == other.fill_value` (line 1727).

This creates asymmetry because:
- `SparseDtype(float32, nan) == SparseDtype(float32, 0.0)`: The NaN side uses type checking, `isinstance(0.0, type(nan))` evaluates to True (both are floats).
- `SparseDtype(float32, 0.0) == SparseDtype(float32, nan)`: The non-NaN side uses value equality, `0.0 == nan` evaluates to False.

The developers added a comment (lines 1708-1712) explaining they wanted to handle different NaN representations as equal while excluding incompatible types like datetime NaT, but the implementation inadvertently made comparisons between NaN and non-NaN values asymmetric.

## Proposed Fix

The fix ensures that both dtypes must have NaN fill values for them to be considered equal when using the type-based comparison:

```diff
diff --git a/pandas/core/dtypes/dtypes.py b/pandas/core/dtypes/dtypes.py
index 1234567..abcdefg 100644
--- a/pandas/core/dtypes/dtypes.py
+++ b/pandas/core/dtypes/dtypes.py
@@ -1713,9 +1713,9 @@ class SparseDtype(ExtensionDtype):
                 fill_value = (
                     other._is_na_fill_value
-                    and isinstance(self.fill_value, type(other.fill_value))
-                    or isinstance(other.fill_value, type(self.fill_value))
+                    and (isinstance(self.fill_value, type(other.fill_value))
+                         or isinstance(other.fill_value, type(self.fill_value)))
                 )
             else:
                 with warnings.catch_warnings():
                     # Ignore spurious numpy warning
```