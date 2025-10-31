# Bug Report: dask.dataframe.io.parquet _DNF.combine Lacks Associativity

**Target**: `dask.dataframe.dask_expr.io.parquet._DNF.combine`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_DNF.combine` method does not satisfy the associativity property, causing `(a.combine(b)).combine(c)` to produce different filter results than `a.combine(b.combine(c))`. This can lead to incorrect query results when filters are combined in different orders.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.dask_expr.io.parquet import _DNF

filter_tuple = st.tuples(
    st.text(min_size=1, max_size=20),
    st.sampled_from([">", "<", ">=", "<=", "==", "!="]),
    st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=20))
)

single_filter = st.one_of(
    filter_tuple,
    st.lists(filter_tuple, min_size=1, max_size=5),
    st.lists(st.lists(filter_tuple, min_size=1, max_size=3), min_size=1, max_size=3)
)

@given(single_filter, single_filter, single_filter)
@settings(max_examples=500)
def test_dnf_combine_associativity(f1, f2, f3):
    dnf1 = _DNF(f1)
    dnf2 = _DNF(f2)
    dnf3 = _DNF(f3)

    combined_12_3 = dnf1.combine(dnf2).combine(dnf3)
    combined_1_23 = dnf1.combine(dnf2.combine(dnf3))

    assert combined_12_3._filters == combined_1_23._filters
```

**Failing input**:
```python
f1 = ('0', '>', 0)
f2 = [[('0', '>', 0)], [('0', '>', 0), ('0', '<', '')], [('0', '<', 0), ('00', '<', '')]]
f3 = [[('0', '>', 0)], [('0', '>', 0), ('0', '<', '')], [('0', '<', 0), ('00', '<', '')]]
```

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.dask_expr.io.parquet import _DNF

f1 = ('a', '>', 5)
f2 = [[('b', '>', 0)], [('c', '<', 10)]]
f3 = [[('d', '>', 0)], [('e', '<', 10)]]

dnf1 = _DNF(f1)
dnf2 = _DNF(f2)
dnf3 = _DNF(f3)

result_12_3 = dnf1.combine(dnf2).combine(dnf3)
result_1_23 = dnf1.combine(dnf2.combine(dnf3))

print(f"(dnf1 & dnf2) & dnf3 = {result_12_3._filters}")
print(f"dnf1 & (dnf2 & dnf3) = {result_1_23._filters}")
print(f"Equal? {result_12_3._filters == result_1_23._filters}")
```

## Why This Is A Bug

The `combine` method is used to combine filter expressions in Disjunctive Normal Form (DNF). The logical AND operation should be associative - that is, `(A AND B) AND C` should always be equivalent to `A AND (B AND C)` for any logical formulas A, B, and C.

The bug occurs in the normalization process (lines 1639-1643 of parquet.py). When an `_And` object containing already-normalized DNF filters is normalized, the Cartesian product expansion can produce different results depending on the nesting structure of the `_And` objects.

This violates a fundamental algebraic property and can cause:
1. Different query results when filters are applied in different orders
2. Inconsistent behavior in query optimization
3. Potential data corruption when wrong rows are filtered

## Fix

The issue is that `normalize` method treats nested `_And` structures differently based on their depth. When combining filters, the method creates `_And([self._filters, other._filters])` at line 1658, but this nested structure is not properly flattened during normalization.

A potential fix would be to ensure that when normalizing an `_And` containing DNF forms (which are `_Or` objects), the Cartesian product is computed consistently. Specifically, the `_And` elements should be flattened before computing the product:

```diff
diff --git a/dask/dataframe/dask_expr/io/parquet.py b/dask/dataframe/dask_expr/io/parquet.py
index 1234567..abcdefg 100644
--- a/dask/dataframe/dask_expr/io/parquet.py
+++ b/dask/dataframe/dask_expr/io/parquet.py
@@ -1636,10 +1636,14 @@ class _DNF:
             result = cls._Or((cls._And((filters,)),))
         elif isinstance(filters, cls._Or):
             result = cls._Or(se for e in filters for se in cls.normalize(e))
         elif isinstance(filters, cls._And):
+            # Normalize each element first to ensure consistent structure
             total = []
-            for c in itertools.product(*[cls.normalize(e) for e in filters]):
-                total.append(cls._And(se for e in c for se in e))
+            normalized_elements = [cls.normalize(e) for e in filters]
+            # Each normalized element is an _Or of _And, we need to distribute
+            for combination in itertools.product(*normalized_elements):
+                # Flatten all _And elements from this combination into one _And
+                flattened = cls._And(term for and_clause in combination for term in and_clause)
+                total.append(flattened)
             result = cls._Or(total)
         else:
             raise TypeError(f"{type(filters)} not a supported type for _DNF")
```

However, a more thorough fix might require restructuring how `combine` constructs the intermediate `_And` to avoid deep nesting altogether.