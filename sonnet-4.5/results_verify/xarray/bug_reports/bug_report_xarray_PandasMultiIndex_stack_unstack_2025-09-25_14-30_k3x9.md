# Bug Report: xarray.core.indexes.PandasMultiIndex stack/unstack roundtrip failure

**Target**: `xarray.core.indexes.PandasMultiIndex.stack()` and `xarray.core.indexes.PandasMultiIndex.unstack()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`PandasMultiIndex.stack()` accepts input variables with duplicate values and successfully creates a MultiIndex with duplicate entries. However, `PandasMultiIndex.unstack()` raises a `ValueError` when attempting to unstack such a MultiIndex, violating the documented stack/unstack roundtrip property.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from xarray.core.indexes import PandasMultiIndex
from xarray.core.variable import Variable

@given(st.lists(st.integers(), min_size=1, max_size=50))
def test_pandas_multi_index_stack_unstack_roundtrip(level_a_values):
    level_b_values = ['a', 'b']

    var_a = Variable(("dim_a",), np.array(level_a_values))
    var_b = Variable(("dim_b",), np.array(level_b_values))

    variables = {"level_a": var_a, "level_b": var_b}

    multi_idx = PandasMultiIndex.stack(variables, "stacked")

    unstacked_indexes, pd_multi_index = multi_idx.unstack()

    assert "level_a" in unstacked_indexes
    assert "level_b" in unstacked_indexes
```

**Failing input**: `level_a_values=[0, 0]`

## Reproducing the Bug

```python
import numpy as np
from xarray.core.indexes import PandasMultiIndex
from xarray.core.variable import Variable

level_a_values = [0, 0]
level_b_values = ['a', 'b']

var_a = Variable(("dim_a",), np.array(level_a_values))
var_b = Variable(("dim_b",), np.array(level_b_values))

variables = {"level_a": var_a, "level_b": var_b}

multi_idx = PandasMultiIndex.stack(variables, "stacked")
print(f"Stacked successfully: {multi_idx.index.tolist()}")

unstacked_indexes, pd_multi_index = multi_idx.unstack()
```

**Output:**
```
Stacked successfully: [(0, 'a'), (0, 'b'), (0, 'a'), (0, 'b')]
Traceback (most recent call last):
  ...
ValueError: Cannot unstack MultiIndex containing duplicates. Make sure entries are unique, e.g., by  calling ``.drop_duplicates('stacked')``, before unstacking.
```

## Why This Is A Bug

The docstring for `PandasMultiIndex.stack()` (line 1092-1093 in `xarray/core/indexes.py`) explicitly states:

> "Keeps levels the same (doesn't refactorize them) so that it gives back the original labels after a stack/unstack roundtrip."

This is a contract violation. When `stack()` is called with variables containing duplicate values (which is valid input according to the method's type signature and documentation), it creates a MultiIndex with duplicate entries. However, `unstack()` cannot handle this case, breaking the documented roundtrip property.

The bug can manifest in real usage when:
1. Users stack coordinate variables that happen to have duplicate values
2. They later attempt to unstack, expecting the documented roundtrip to work
3. The operation fails with a confusing error message

## Fix

The fix should either:

**Option 1: Validate inputs in `stack()` (Recommended)**

Add validation to ensure the cartesian product of input variables will not create duplicates:

```diff
diff --git a/xarray/core/indexes.py b/xarray/core/indexes.py
index 1234567..abcdefg 100644
--- a/xarray/core/indexes.py
+++ b/xarray/core/indexes.py
@@ -1095,6 +1095,16 @@ class PandasMultiIndex(PandasIndex):
         """
         _check_dim_compat(variables, all_dims="different")

         level_indexes = [safe_cast_to_index(var) for var in variables.values()]
+
+        # Check that the cartesian product won't create duplicates
+        # This ensures stack/unstack roundtrip works as documented
+        import itertools
+        product = list(itertools.product(*[idx.tolist() for idx in level_indexes]))
+        if len(product) != len(set(product)):
+            raise ValueError(
+                f"cannot create a multi-index along stacked dimension {dim!r} "
+                "because the cartesian product of the input variables contains duplicates. "
+                "Ensure all combinations of level values are unique."
+            )
+
         for name, idx in zip(variables, level_indexes, strict=True):
             if isinstance(idx, pd.MultiIndex):
```

**Option 2: Update documentation**

If duplicate MultiIndex entries are intentionally supported by `stack()`, update the docstring to clarify that the roundtrip only works for inputs that produce unique MultiIndex entries, and update `unstack()` to handle this case or provide a clearer error message.