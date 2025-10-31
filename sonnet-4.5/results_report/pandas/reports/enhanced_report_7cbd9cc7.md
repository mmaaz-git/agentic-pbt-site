# Bug Report: pandas.io.common.dedup_names AssertionError with Non-Tuple Duplicates

**Target**: `pandas.io.common.dedup_names`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `dedup_names` function crashes with an `AssertionError` when called with duplicate non-tuple elements and `is_potential_multiindex=True`, despite its type signature explicitly accepting any `Sequence[Hashable]`.

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

if __name__ == "__main__":
    test_dedup_names_multiindex_with_non_tuples_and_duplicates()
```

<details>

<summary>
**Failing input**: `['0', '0']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 17, in <module>
    test_dedup_names_multiindex_with_non_tuples_and_duplicates()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 5, in test_dedup_names_multiindex_with_non_tuples_and_duplicates
    names=st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=20)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/53/hypo.py", line 10, in test_dedup_names_multiindex_with_non_tuples_and_duplicates
    result = pd_common.dedup_names(names, is_potential_multiindex=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 1258, in dedup_names
    assert isinstance(col, tuple)
           ~~~~~~~~~~^^^^^^^^^^^^
AssertionError
Falsifying example: test_dedup_names_multiindex_with_non_tuples_and_duplicates(
    names=['0', '0'],
)
```
</details>

## Reproducing the Bug

```python
import pandas.io.common as pd_common
import sys

# Demonstrate the bug: calling dedup_names with duplicate strings
# and is_potential_multiindex=True causes an AssertionError
names = ['0', '0']
print(f"Input names: {names}")
print(f"is_potential_multiindex: True")
print("\nAttempting to call dedup_names...")
sys.stdout.flush()

result = pd_common.dedup_names(names, is_potential_multiindex=True)
print(f"Result: {result}")
```

<details>

<summary>
AssertionError at line 1258 in dedup_names
</summary>
```
Input names: ['0', '0']
is_potential_multiindex: True

Attempting to call dedup_names...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/53/repo.py", line 12, in <module>
    result = pd_common.dedup_names(names, is_potential_multiindex=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py", line 1258, in dedup_names
    assert isinstance(col, tuple)
           ~~~~~~~~~~^^^^^^^^^^^^
AssertionError
```
</details>

## Why This Is A Bug

The `dedup_names` function violates its own type contract in multiple ways:

1. **Type signature promises broad compatibility**: The function signature `dedup_names(names: Sequence[Hashable], is_potential_multiindex: bool) -> Sequence[Hashable]` explicitly accepts any `Hashable` type (strings, integers, tuples, etc.), not exclusively tuples.

2. **Assertion is misused for runtime enforcement**: Line 1258 contains `assert isinstance(col, tuple)` with the comment `# for mypy`, indicating this was intended for static type checking only. However, assertions cause runtime crashes in Python, making this a contract violation when non-tuple duplicates are encountered.

3. **Inconsistent behavior exposes implementation details**: The function successfully processes non-tuple inputs when there are no duplicates (e.g., `['a', 'b', 'c']` with `is_potential_multiindex=True` works fine). The assertion is only triggered when entering the duplicate-handling loop (line 1253: `while cur_count > 0`), creating an inconsistent and confusing API.

4. **Documentation provides no warning**: The docstring shows only an example with `is_potential_multiindex=False` and makes no mention of tuple requirements for the `True` case. Users have no way to know about this hidden constraint.

## Relevant Context

The `dedup_names` function is an internal pandas utility used primarily by data parsers to handle duplicate column names. When `is_potential_multiindex=True`, it's meant to handle MultiIndex columns (which are tuples).

The related function `is_potential_multi_index` (lines 1226-1230) checks if all columns are tuples before setting `is_potential_multiindex=True`. In normal pandas usage through parsers, non-tuple inputs with `is_potential_multiindex=True` shouldn't occur. However, since the function is importable and has a public-looking interface, it should either:
- Handle non-tuples gracefully
- Provide clear error messages instead of assertions

The function is located at `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/common.py` starting at line 1233.

## Proposed Fix

```diff
--- a/pandas/io/common.py
+++ b/pandas/io/common.py
@@ -1254,8 +1254,10 @@ def dedup_names(
         while cur_count > 0:
             counts[col] = cur_count + 1

             if is_potential_multiindex:
-                # for mypy
-                assert isinstance(col, tuple)
+                if not isinstance(col, tuple):
+                    raise TypeError(
+                        f"When is_potential_multiindex=True, column names must be tuples, got {type(col).__name__}: {col!r}"
+                    )
                 col = col[:-1] + (f"{col[-1]}.{cur_count}",)
             else:
                 col = f"{col}.{cur_count}"
```