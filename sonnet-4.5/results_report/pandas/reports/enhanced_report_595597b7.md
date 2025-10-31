# Bug Report: pandas.io.parsers._validate_names Raises Wrong Exception Type

**Target**: `pandas.io.parsers.readers._validate_names`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_validate_names` function raises `TypeError` instead of the documented `ValueError` when given unhashable column names like lists or dictionaries, violating its API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import io

@given(
    unhashable_names=st.lists(
        st.one_of(
            st.lists(st.integers(), min_size=1, max_size=3),
            st.dictionaries(st.text(min_size=1, max_size=3), st.integers(), min_size=1, max_size=2)
        ),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=50)
def test_unhashable_names_should_raise_valueerror(unhashable_names):
    csv_data = ','.join(['0'] * len(unhashable_names)) + '\n'
    try:
        df = pd.read_csv(io.StringIO(csv_data), names=unhashable_names, header=None)
        assert False, "Should have raised an error"
    except TypeError:
        assert False, "Got TypeError but docstring promises ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_unhashable_names_should_raise_valueerror()
```

<details>

<summary>
**Failing input**: `unhashable_names=[[0], [0]]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 19, in test_unhashable_names_should_raise_valueerror
    df = pd.read_csv(io.StringIO(csv_data), names=unhashable_names, header=None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/parsers/readers.py", line 617, in _read
    _validate_names(kwds.get("names", None))
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/parsers/readers.py", line 575, in _validate_names
    if len(names) != len(set(names)):
                         ~~~^^^^^^^
TypeError: unhashable type: 'list'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 27, in <module>
    test_unhashable_names_should_raise_valueerror()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 6, in test_unhashable_names_should_raise_valueerror
    unhashable_names=st.lists(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 22, in test_unhashable_names_should_raise_valueerror
    assert False, "Got TypeError but docstring promises ValueError"
           ^^^^^
AssertionError: Got TypeError but docstring promises ValueError
Falsifying example: test_unhashable_names_should_raise_valueerror(
    unhashable_names=[[0], [0]],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import io

csv_data = "1,2,3\n4,5,6"
names = [[1, 2], [3, 4], [5, 6]]

try:
    df = pd.read_csv(io.StringIO(csv_data), names=names, header=None)
except TypeError as e:
    print(f"BUG: Raised {type(e).__name__}: {e}")
    print("Expected: ValueError according to docstring")
except ValueError as e:
    print(f"Correct: Raised {type(e).__name__}: {e}")
```

<details>

<summary>
BUG: Raised TypeError instead of documented ValueError
</summary>
```
BUG: Raised TypeError: unhashable type: 'list'
Expected: ValueError according to docstring
```
</details>

## Why This Is A Bug

The `_validate_names` function in pandas/io/parsers/readers.py violates its documented API contract in multiple ways:

1. **Docstring Promise (lines 561-562)**: The docstring explicitly states "Raise ValueError if the `names` parameter contains duplicates or has an invalid data type." The function promises to raise `ValueError` for invalid data types, but raises `TypeError` when encountering unhashable elements.

2. **Type Annotation Contract (line 559)**: The function signature declares `def _validate_names(names: Sequence[Hashable] | None) -> None:`, indicating it expects a sequence of hashable elements. When this contract is violated, the function should validate and raise the documented `ValueError`, not allow Python's internal `TypeError` to bubble up.

3. **Semantic Correctness**: The distinction between `ValueError` and `TypeError` matters for API consistency. `ValueError` is appropriate here because:
   - The input type is correct (it IS a sequence)
   - The values within the sequence are invalid (non-hashable when hashable is required)
   - This is similar to how a function expecting positive integers would raise `ValueError` (not `TypeError`) for negative integers

4. **Exception Handling**: Code that relies on the documented behavior may have `try-except` blocks catching `ValueError` as promised. The current implementation breaks this contract by raising an undocumented exception type.

The root cause is on line 575 where `set(names)` is called before validating that elements are hashable. Python's `set()` constructor raises `TypeError` for unhashable elements before the function can perform its documented validation.

## Relevant Context

- This is an internal validation function used by `pd.read_csv()` and related parsing functions
- The pandas documentation for `read_csv` confirms that 'names' should be a "Sequence of Hashable"
- While passing unhashable types as column names is uncommon, maintaining API consistency is important for reliability
- The current error message "unhashable type: 'list'" is informative but violates the documented contract
- Function source: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/parsers/readers.py:559-580`

## Proposed Fix

```diff
--- a/pandas/io/parsers/readers.py
+++ b/pandas/io/parsers/readers.py
@@ -572,9 +572,15 @@ def _validate_names(names: Sequence[Hashable] | None) -> None:
         If names are not unique or are not ordered (e.g. set).
     """
     if names is not None:
-        if len(names) != len(set(names)):
-            raise ValueError("Duplicate names are not allowed.")
         if not (
             is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)
         ):
             raise ValueError("Names should be an ordered collection.")
+        try:
+            if len(names) != len(set(names)):
+                raise ValueError("Duplicate names are not allowed.")
+        except TypeError:
+            raise ValueError(
+                "Column names must be hashable. "
+                "Unhashable types like list, dict, or set are not allowed."
+            ) from None
```