# Bug Report: scipy.sparse.csgraph._validation.validate_graph ignores dtype parameter

**Target**: `scipy.sparse.csgraph._validation.validate_graph`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validate_graph` function accepts a `dtype` parameter but ignores it completely, always using the module-level `DTYPE` constant (`np.float64`) instead. This violates the function's API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from scipy.sparse.csgraph._validation import validate_graph

@given(st.sampled_from([np.float32, np.float64, np.int32]))
def test_validate_graph_respects_dtype(dtype):
    G = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    result = validate_graph(G, directed=True, dtype=dtype)

    assert result.dtype == dtype, \
        f"Expected dtype {dtype}, but got {result.dtype}"
```

**Failing input**: Any dtype other than `np.float64`

## Reproducing the Bug

```python
import numpy as np
from scipy.sparse.csgraph._validation import validate_graph

G = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.int32)

result_float32 = validate_graph(G, directed=True, dtype=np.float32)
result_float64 = validate_graph(G, directed=True, dtype=np.float64)

print(f"Result with dtype=np.float32: {result_float32.dtype}")
print(f"Result with dtype=np.float64: {result_float64.dtype}")

assert result_float32.dtype == np.float32
```

Output:
```
Result with dtype=np.float32: float64
Result with dtype=np.float64: float64
AssertionError
```

## Why This Is A Bug

The function signature declares `dtype=DTYPE` as a parameter, indicating users can control the output dtype. However, the implementation always uses the module-level `DTYPE` constant instead of the parameter value.

Evidence from `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/sparse/csgraph/_validation.py`:

- Line 35: `csgraph.tocsr(copy=copy_if_sparse).astype(DTYPE, copy=False)` - uses `DTYPE` not `dtype`
- Line 41: `np.array(csgraph.data, dtype=DTYPE, copy=copy_if_dense)` - uses `DTYPE` not `dtype`
- Line 53: `np.asarray(csgraph.data, dtype=DTYPE)` - uses `DTYPE` not `dtype`

This is an API contract violation - the parameter exists but has no effect.

## Fix

Replace all occurrences of the module-level `DTYPE` constant with the `dtype` parameter within the function:

```diff
--- a/scipy/sparse/csgraph/_validation.py
+++ b/scipy/sparse/csgraph/_validation.py
@@ -32,7 +32,7 @@ def validate_graph(csgraph, directed, dtype=DTYPE,

     if issparse(csgraph):
         if csr_output:
-            csgraph = csgraph.tocsr(copy=copy_if_sparse).astype(DTYPE, copy=False)
+            csgraph = csgraph.tocsr(copy=copy_if_sparse).astype(dtype, copy=False)
         else:
             csgraph = csgraph_to_dense(csgraph, null_value=null_value_out)
     elif np.ma.isMaskedArray(csgraph):
@@ -38,7 +38,7 @@ def validate_graph(csgraph, directed, dtype=DTYPE,
     elif np.ma.isMaskedArray(csgraph):
         if dense_output:
             mask = csgraph.mask
-            csgraph = np.array(csgraph.data, dtype=DTYPE, copy=copy_if_dense)
+            csgraph = np.array(csgraph.data, dtype=dtype, copy=copy_if_dense)
             csgraph[mask] = null_value_out
         else:
             csgraph = csgraph_from_masked(csgraph)
@@ -50,7 +50,7 @@ def validate_graph(csgraph, directed, dtype=DTYPE,
                                                 infinity_null=infinity_null)
             mask = csgraph.mask
-            csgraph = np.asarray(csgraph.data, dtype=DTYPE)
+            csgraph = np.asarray(csgraph.data, dtype=dtype)
             csgraph[mask] = null_value_out
         else:
             csgraph = csgraph_from_dense(csgraph, null_value=null_value_in,
```