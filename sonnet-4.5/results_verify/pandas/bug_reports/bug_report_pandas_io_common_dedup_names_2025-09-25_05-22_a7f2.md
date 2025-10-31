# Bug Report: pandas.io.common.dedup_names AssertionError with non-tuple duplicates

**Target**: `pandas.io.common.dedup_names`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `dedup_names` function has a type signature that accepts `Sequence[Hashable]`, but raises an `AssertionError` when called with duplicate non-tuple elements and `is_potential_multiindex=True`. This violates the function's documented interface.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import pandas.io.common as pd_common

@given(
    names=st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=20)
)
def test_dedup_names_multiindex_with_non_tuples_and_duplicates(names):
    assume(len(names) != len(set(names)))

    result = pd_common.dedup_names(names, is_potential_multiindex=True)
    result_list = list(result)

    assert len(result_list) == len(names)
    assert len(result_list) == len(set(result_list))
```

**Failing input**: `['0', '0']`

## Reproducing the Bug

```python
import pandas.io.common as pd_common

names = ['0', '0']
result = pd_common.dedup_names(names, is_potential_multiindex=True)
```

## Why This Is A Bug

1. **Type signature mismatch**: The function signature is `dedup_names(names: Sequence[Hashable], is_potential_multiindex: bool) -> Sequence[Hashable]`, which accepts any `Hashable` type (strings, integers, etc.), not just tuples.

2. **No documentation of tuple requirement**: The docstring does not indicate that tuples are required when `is_potential_multiindex=True`. The only example shown uses strings with `is_potential_multiindex=False`.

3. **Runtime assertion instead of type checking**: The assertion `assert isinstance(col, tuple)` with the comment `# for mypy` suggests this is meant for static type checking, but it causes a runtime crash. Type constraints should be enforced through type annotations or proper validation with helpful error messages.

4. **Inconsistent behavior**: The function works fine with non-tuple inputs when there are no duplicates (the assertion is never reached), but fails as soon as duplicates are encountered.

## Fix

The function should either handle non-tuple inputs gracefully or provide a clear error message. Here's a patch that adds proper type checking with a helpful error:

```diff
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -1255,8 +1255,10 @@ def dedup_names(
         while cur_count > 0:
             counts[col] = cur_count + 1

             if is_potential_multiindex:
-                # for mypy
-                assert isinstance(col, tuple)
+                if not isinstance(col, tuple):
+                    raise TypeError(
+                        f"When is_potential_multiindex=True, column names must be tuples, got {type(col).__name__}"
+                    )
                 col = col[:-1] + (f"{col[-1]}.{cur_count}",)
             else:
                 col = f"{col}.{cur_count}"
```