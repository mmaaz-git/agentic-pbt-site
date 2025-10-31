# Bug Report: numpy.ma.allequal fill_value=False Ignores Unmasked Values

**Target**: `numpy.ma.allequal`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.ma.allequal(a, b, fill_value=False)` incorrectly returns `False` without comparing unmasked values when arrays contain any masked elements, even when all unmasked values and masks are identical.

## Property-Based Test

```python
import numpy as np
import numpy.ma as ma
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps


@st.composite
def identical_masked_arrays_with_some_masked(draw):
    size = draw(st.integers(min_value=2, max_value=30))
    data = draw(nps.arrays(dtype=np.float64, shape=(size,),
                          elements={"allow_nan": False, "allow_infinity": False,
                                   "min_value": -100, "max_value": 100}))
    mask = draw(nps.arrays(dtype=bool, shape=(size,)))

    assume(mask.any())
    assume((~mask).any())

    return data, mask


@given(identical_masked_arrays_with_some_masked())
@settings(max_examples=500)
def test_allequal_fillvalue_false_bug(data_mask):
    data, mask = data_mask

    x = ma.array(data, mask=mask)
    y = ma.array(data.copy(), mask=mask.copy())

    result_false = ma.allequal(x, y, fill_value=False)

    unmasked_equal = np.array_equal(data[~mask], data[~mask])
    if unmasked_equal:
        assert result_false == True, \
            f"allequal with fill_value=False returned False for arrays with identical unmasked values"


if __name__ == "__main__":
    test_allequal_fillvalue_false_bug()
```

<details>

<summary>
**Failing input**: `data_mask=(array([0., 0.]), array([False, True]))`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 39, in <module>
    test_allequal_fillvalue_false_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 23, in test_allequal_fillvalue_false_bug
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 34, in test_allequal_fillvalue_false_bug
    assert result_false == True, \
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: allequal with fill_value=False returned False for arrays with identical unmasked values
Falsifying example: test_allequal_fillvalue_false_bug(
    data_mask=(array([0., 0.]),
     array([False,  True])),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.ma as ma

# Create two masked arrays with identical unmasked values
x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])

print("Arrays:")
print(f"x = {x}")
print(f"y = {y}")
print()

print("Unmasked values comparison:")
print(f"x unmasked values: {ma.compressed(x)}")
print(f"y unmasked values: {ma.compressed(y)}")
print(f"Unmasked values are identical: {np.array_equal(ma.compressed(x), ma.compressed(y))}")
print()

print("allequal results:")
result_true = ma.allequal(x, y, fill_value=True)
print(f"ma.allequal(x, y, fill_value=True): {result_true}")

result_false = ma.allequal(x, y, fill_value=False)
print(f"ma.allequal(x, y, fill_value=False): {result_false}")
print()

print("Expected behavior:")
print("Since unmasked values are identical and masks are identical,")
print("the function should return True when comparing unmasked values.")
print("With fill_value=False, it incorrectly returns False without checking unmasked values.")
```

<details>

<summary>
Output showing incorrect behavior
</summary>
```
Arrays:
x = [1.0 -- 3.0]
y = [1.0 -- 3.0]

Unmasked values comparison:
x unmasked values: [1. 3.]
y unmasked values: [1. 3.]
Unmasked values are identical: True

allequal results:
ma.allequal(x, y, fill_value=True): True
ma.allequal(x, y, fill_value=False): False

Expected behavior:
Since unmasked values are identical and masks are identical,
the function should return True when comparing unmasked values.
With fill_value=False, it incorrectly returns False without checking unmasked values.
```
</details>

## Why This Is A Bug

The `fill_value` parameter is documented to control "Whether masked values in a or b are considered equal (True) or not (False)". This clearly indicates that `fill_value` should affect how masked positions are treated in the comparison, not whether comparison occurs at all.

The current implementation at line 8461 in `/numpy/ma/core.py` contains a fundamental logic error: when `fill_value=False` and any masks exist, it immediately returns `False` without performing any comparison:

```python
elif fill_value:
    # ... comparison logic ...
else:
    return False  # Bug: bypasses all comparison
```

This violates the function's core purpose - to compare array values for equality. The documentation example shows that the function should still compare unmasked values and return a meaningful result based on that comparison, with `fill_value` only affecting how masked positions contribute to the result.

The bug makes `fill_value=False` essentially useless for masked arrays, reducing it to a simple "return False if any masks exist" which provides no practical value and contradicts both the documentation and user expectations.

## Relevant Context

The numpy.ma.allequal function is part of NumPy's masked array module, which is widely used in scientific computing for handling missing or invalid data. The function is designed to compare two arrays element-wise while properly handling masked values.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.ma.allequal.html

The source code is located at: `/numpy/ma/core.py:8405-8461`

This bug affects any code that relies on `fill_value=False` to perform strict equality checks on masked arrays. While users can work around this by using `fill_value=True` (the default), this doesn't provide the same semantic meaning - sometimes users need to know that masked positions are NOT equal.

## Proposed Fix

```diff
--- a/numpy/ma/core.py
+++ b/numpy/ma/core.py
@@ -8457,8 +8457,14 @@ def allequal(a, b, fill_value=True):
         d = umath.equal(x, y)
         dm = array(d, mask=m, copy=False)
         return dm.filled(True).all(None)
     else:
-        return False
+        x = getdata(a)
+        y = getdata(b)
+        d = umath.equal(x, y)
+        # Check if unmasked values are equal
+        unmasked_equal = d[~m].all() if (~m).any() else True
+        # Check if masks are identical
+        masks_equal = np.array_equal(getmask(a), getmask(b))
+        return unmasked_equal and masks_equal
```