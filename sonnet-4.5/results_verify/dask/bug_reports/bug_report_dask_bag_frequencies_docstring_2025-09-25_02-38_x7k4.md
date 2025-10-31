# Bug Report: dask.bag frequencies() docstring syntax error

**Target**: `dask.bag.Bag.frequencies`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `Bag.frequencies()` contains invalid Python syntax in its example output, using a comma instead of a colon in a dictionary literal.

## Property-Based Test

While exploring dask.bag for property-based testing, I examined the `frequencies()` method to understand its documented behavior for testing the property: `sum(bag.frequencies().values()) == bag.count()`.

## Reproducing the Bug

The bug is in the source code documentation at:
`/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/bag/core.py:938`

```python
def frequencies(self, split_every=None, sort=False):
    """Count number of occurrences of each distinct element.

    >>> import dask.bag as db
    >>> b = db.from_sequence(['Alice', 'Bob', 'Alice'])
    >>> dict(b.frequencies())       # doctest: +SKIP
    {'Alice': 2, 'Bob', 1}
    """
```

The output `{'Alice': 2, 'Bob', 1}` is invalid Python syntax - it should use a colon after 'Bob', not a comma.

## Why This Is A Bug

This violates the contract between code and documentation. The docstring shows invalid Python syntax that would cause a `SyntaxError` if actually executed. This misleads users learning the API.

## Fix

```diff
--- a/dask/bag/core.py
+++ b/dask/bag/core.py
@@ -935,7 +935,7 @@ class Bag(DaskMethodsMixin):
         >>> import dask.bag as db
         >>> b = db.from_sequence(['Alice', 'Bob', 'Alice'])
         >>> dict(b.frequencies())       # doctest: +SKIP
-        {'Alice': 2, 'Bob', 1}
+        {'Alice': 2, 'Bob': 1}
         """
         result = self.reduction(
             frequencies,
```