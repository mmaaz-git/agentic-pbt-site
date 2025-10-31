# Bug Report: dask.dataframe.utils._maybe_sort Permanently Mutates Index Names When They Overlap With Column Names

**Target**: `dask.dataframe.utils._maybe_sort`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_maybe_sort` function permanently renames DataFrame index names to avoid conflicts with column names during sorting, but never restores the original names. This causes unexpected mutations that propagate to callers, including the widely-used `assert_eq` testing utility.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from dask.dataframe.utils import _maybe_sort

@given(st.lists(st.integers(), min_size=2, max_size=10))
def test_maybe_sort_preserves_index_names(data):
    df = pd.DataFrame({'A': data}, index=pd.Index(range(len(data)), name='A'))
    original_name = df.index.names[0]

    result = _maybe_sort(df, check_index=True)

    assert result.index.names[0] == original_name, \
        f"Index name changed from {original_name} to {result.index.names[0]}"

if __name__ == "__main__":
    test_maybe_sort_preserves_index_names()
```

<details>

<summary>
**Failing input**: `data=[0, 0]` (or any list where a DataFrame is created with index.name='A' and column 'A')
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 16, in <module>
    test_maybe_sort_preserves_index_names()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 6, in test_maybe_sort_preserves_index_names
    def test_maybe_sort_preserves_index_names(data):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/4/hypo.py", line 12, in test_maybe_sort_preserves_index_names
    assert result.index.names[0] == original_name, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Index name changed from A to -overlapped-index-name-0
Falsifying example: test_maybe_sort_preserves_index_names(
    data=[0, 0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from dask.dataframe.utils import _maybe_sort

df = pd.DataFrame(
    {'A': [2, 1], 'B': [4, 3]},
    index=pd.Index([10, 20], name='A')
)

print(f"Before: df.index.names = {df.index.names}")

result = _maybe_sort(df, check_index=True)

print(f"After: result.index.names = {result.index.names}")
```

<details>

<summary>
Output shows permanent index name mutation
</summary>
```
Before: df.index.names = ['A']
After: result.index.names = ['-overlapped-index-name-0']
```
</details>

## Why This Is A Bug

The function violates the principle of least surprise by permanently modifying index names as an undocumented side effect. The code at lines 501-505 in `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/utils.py` shows:

1. When index names overlap with column names, the function renames them to avoid conflicts during sorting
2. The renaming uses a pattern `-overlapped-index-name-%d` to ensure uniqueness
3. After sorting by columns (line 505), the function returns without restoring the original names
4. This mutation is visible to all callers, including `assert_eq` (lines 565-566)

The renaming appears designed as a temporary workaround to enable `sort_values(by=columns)` to work when index and column names overlap. However, the implementation makes this change permanent rather than temporary, causing unexpected behavior in testing utilities and potentially masking bugs in test suites.

## Relevant Context

- The `_maybe_sort` function is an internal utility (underscore prefix indicates private)
- It's primarily used by `assert_eq`, a critical testing function in Dask's test suite
- The `assert_eq` function uses `_maybe_sort` when `sort_results=True` (default behavior)
- GitHub issue #9018 acknowledges that `assert_eq` sorting can mask issues in tests
- The function has minimal documentation - only a comment "# sort by value, then index"
- No existing documentation specifies whether index names should be preserved

Relevant code location: https://github.com/dask/dask/blob/main/dask/dataframe/utils.py#L495-L510

## Proposed Fix

```diff
--- a/dask/dataframe/utils.py
+++ b/dask/dataframe/utils.py
@@ -498,10 +498,14 @@ def _maybe_sort(a, check_index: bool):
     # sort by value, then index
     try:
         if is_dataframe_like(a):
+            original_index_names = None
             if set(a.index.names) & set(a.columns):
+                original_index_names = a.index.names
                 a.index.names = [
                     "-overlapped-index-name-%d" % i for i in range(len(a.index.names))
                 ]
             a = a.sort_values(by=methods.tolist(a.columns))
+            if original_index_names is not None:
+                a.index.names = original_index_names
         else:
             a = a.sort_values()
     except (TypeError, IndexError, ValueError):
```