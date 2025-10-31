# Bug Report: pandas.io.common.dedup_names Crashes with Non-Tuple Columns

**Target**: `pandas.io.common.dedup_names`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `dedup_names` function crashes with an `AssertionError` when `is_potential_multiindex=True` is passed with non-tuple column names, despite accepting any boolean value for the parameter and not documenting this precondition.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.common as common


@given(st.lists(st.text()), st.booleans())
def test_dedup_names_preserves_length(names, is_potential_multiindex):
    result = common.dedup_names(names, is_potential_multiindex)
    assert len(result) == len(names)
```

**Failing input**: `names=['', ''], is_potential_multiindex=True`

## Reproducing the Bug

```python
from pandas.io.common import dedup_names

dedup_names(['', ''], is_potential_multiindex=True)

dedup_names(['x', 'x'], is_potential_multiindex=True)

dedup_names(['a', 'b', 'c'], is_potential_multiindex=True)
```

## Why This Is A Bug

1. **Undocumented precondition**: The function signature accepts `is_potential_multiindex: bool` without documenting that when `True`, all column names must be tuples.

2. **Assertion instead of validation**: The code uses `assert isinstance(col, tuple)` for type checking (with comment "for mypy"), but assertions can be disabled with Python's `-O` flag and should not be used for input validation.

3. **Inconsistent with API design**: The function accepts a boolean parameter that users might set based on their understanding of their data structure, but crashes instead of validating or handling the mismatch.

4. **No helpful error message**: Users get a bare `AssertionError` with no explanation of what went wrong or how to fix it.

The function should either:
- Validate inputs and raise a `ValueError` with a clear message
- Handle non-tuple columns gracefully when `is_potential_multiindex=True`
- Document the precondition clearly in the docstring

## Fix

```diff
def dedup_names(
    names: Sequence[Hashable], is_potential_multiindex: bool
) -> Sequence[Hashable]:
    """
    Rename column names if duplicates exist.

    Currently the renaming is done by appending a period and an autonumeric,
    but a custom pattern may be supported in the future.

+   Parameters
+   ----------
+   names : Sequence[Hashable]
+       Column names to deduplicate
+   is_potential_multiindex : bool
+       If True, assumes column names are tuples (for MultiIndex columns)
+
    Examples
    --------
    >>> dedup_names(["x", "y", "x", "x"], is_potential_multiindex=False)
    ['x', 'y', 'x.1', 'x.2']
    """
    names = list(names)  # so we can index
    counts: DefaultDict[Hashable, int] = defaultdict(int)

    for i, col in enumerate(names):
        cur_count = counts[col]

        while cur_count > 0:
            counts[col] = cur_count + 1

            if is_potential_multiindex:
-               # for mypy
-               assert isinstance(col, tuple)
+               if not isinstance(col, tuple):
+                   raise ValueError(
+                       f"When is_potential_multiindex=True, all column names must be tuples. "
+                       f"Got {type(col).__name__} at index {i}: {col!r}"
+                   )
                col = col[:-1] + (f"{col[-1]}.{cur_count}",)
            else:
                col = f"{col}.{cur_count}"
            cur_count = counts[col]

        names[i] = col
        counts[col] = cur_count + 1

    return names
```