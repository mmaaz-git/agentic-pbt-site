# Bug Report: pandas.core.internals.concat._is_uniform_join_units Asymmetry

**Target**: `pandas.core.internals.concat._is_uniform_join_units`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_is_uniform_join_units` function in pandas.core.internals.concat exhibits asymmetric behavior when checking if join units are uniform. The result depends on the order of blocks, violating the expected symmetry property for uniformity checks.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas._libs.internals import BlockPlacement
from pandas.core.internals.api import make_block
from pandas.core.internals.concat import _is_uniform_join_units, JoinUnit


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=50)
def test_uniform_join_units_should_be_symmetric(ncols):
    values_float = np.random.rand(1, ncols).astype(np.float64)
    values_int = np.random.randint(0, 100, (1, ncols)).astype(np.int64)

    placement = BlockPlacement(slice(0, 1))

    block_float = make_block(values_float, placement, ndim=2)
    block_int = make_block(values_int, placement, ndim=2)

    join_units_fi = [JoinUnit(block_float), JoinUnit(block_int)]
    join_units_if = [JoinUnit(block_int), JoinUnit(block_float)]

    result_fi = _is_uniform_join_units(join_units_fi)
    result_if = _is_uniform_join_units(join_units_if)

    assert result_fi == result_if, (
        f"_is_uniform_join_units should be symmetric: "
        f"[float64, int64] = {result_fi}, "
        f"[int64, float64] = {result_if}"
    )
```

**Failing input**: `ncols=1` (or any value)

## Reproducing the Bug

```python
import numpy as np
from pandas._libs.internals import BlockPlacement
from pandas.core.internals.api import make_block
from pandas.core.internals.concat import _is_uniform_join_units, JoinUnit

values_float = np.array([[1.0]], dtype=np.float64)
values_int = np.array([[1]], dtype=np.int64)
placement = BlockPlacement(slice(0, 1))

block_float = make_block(values_float, placement, ndim=2)
block_int = make_block(values_int, placement, ndim=2)

join_units_fi = [JoinUnit(block_float), JoinUnit(block_int)]
join_units_if = [JoinUnit(block_int), JoinUnit(block_float)]

result_fi = _is_uniform_join_units(join_units_fi)
result_if = _is_uniform_join_units(join_units_if)

print(f"[float64, int64]: {result_fi}")
print(f"[int64, float64]: {result_if}")
```

Output:
```
[float64, int64]: True
[int64, float64]: False
```

## Why This Is A Bug

The `_is_uniform_join_units` function should determine if a list of join units are uniform in a symmetric way - the order shouldn't matter. However, the current implementation produces different results based on block order:
- `[float64, int64]` returns `True`
- `[int64, float64]` returns `False`

This asymmetry violates the mathematical property that uniformity should be commutative. The root cause is in lines 587-592 of concat.py:

```python
all(
    ju.block.dtype == first.dtype
    or ju.block.dtype.kind in "iub"
    for ju in join_units
)
```

The condition `ju.block.dtype.kind in "iub"` allows any block with integer/unsigned/boolean kind to pass, regardless of the first block's dtype. This creates asymmetry:
- When first is float64 and second is int64: both conditions pass (float64==float64, int64.kind in "iub")
- When first is int64 and second is float64: second fails (float64!=int64 and float64.kind not in "iub")

## Fix

```diff
--- a/pandas/core/internals/concat.py
+++ b/pandas/core/internals/concat.py
@@ -585,7 +585,7 @@ def _is_uniform_join_units(join_units: list[JoinUnit]) -> bool:
         # e.g. DatetimeLikeBlock can be dt64 or td64, but these are not uniform
         all(
             ju.block.dtype == first.dtype
             # GH#42092 we only want the dtype_equal check for non-numeric blocks
             #  (for now, may change but that would need a deprecation)
-            or ju.block.dtype.kind in "iub"
+            or (first.dtype.kind in "iub" and ju.block.dtype.kind in "iub")
             for ju in join_units
         )
         and
```

This fix ensures that the relaxed dtype checking only applies when BOTH the first block and the current block are numeric (kind in "iub"), making the check symmetric.