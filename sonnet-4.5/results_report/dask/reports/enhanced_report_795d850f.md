# Bug Report: dask.dataframe.io.orc._read_orc Mutates Input Columns Parameter

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The internal function `_read_orc` mutates its `columns` parameter by appending the index column to it, violating the principle that functions should not have unexpected side effects on input parameters.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pyarrow as pa
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from dask.dataframe.io.orc.core import _read_orc


@given(
    st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)), min_size=1, max_size=5, unique=True),
    st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122))
)
@settings(max_examples=100)
def test_read_orc_does_not_mutate_columns(column_names, index_name):
    columns_list = list(column_names)
    original_columns = columns_list.copy()

    parts = [("dummy_path", [0])]
    engine = ArrowORCEngine
    fs = None
    schema = {col: pa.int64() for col in column_names}

    try:
        _read_orc(parts, engine=engine, fs=fs, schema=schema, index=index_name, columns=columns_list)
    except:
        pass

    assert columns_list == original_columns, f"columns list was mutated: {original_columns} -> {columns_list}"

if __name__ == "__main__":
    test_read_orc_does_not_mutate_columns()
```

<details>

<summary>
**Failing input**: `column_names=['a'], index_name='a'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 29, in <module>
    test_read_orc_does_not_mutate_columns()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 8, in test_read_orc_does_not_mutate_columns
    st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)), min_size=1, max_size=5, unique=True),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 26, in test_read_orc_does_not_mutate_columns
    assert columns_list == original_columns, f"columns list was mutated: {original_columns} -> {columns_list}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: columns list was mutated: ['a'] -> ['a', 'a']
Falsifying example: test_read_orc_does_not_mutate_columns(
    # The test always failed when commented parts were varied together.
    column_names=['a'],  # or any other generated value
    index_name='a',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import pyarrow as pa

# Create a minimal test case showing the mutation bug
columns_list = ['col1', 'col2']
original = columns_list.copy()

print(f"Original columns list before call: {columns_list}")

# Set up dummy parameters for _read_orc
parts = [("dummy_path", [0])]
schema = {'col1': pa.int64(), 'col2': pa.int64()}

# Call _read_orc - this will fail because dummy_path doesn't exist,
# but the mutation happens before the file read
try:
    _read_orc(
        parts,
        engine=ArrowORCEngine,
        fs=None,
        schema=schema,
        index='col1',
        columns=columns_list
    )
except Exception as e:
    print(f"Expected error (file doesn't exist): {e}")

print(f"\nAfter calling _read_orc:")
print(f"Original copy:   {original}")
print(f"Modified list:   {columns_list}")
print(f"Lists are equal: {columns_list == original}")
print(f"Mutation occurred: {columns_list != original}")

# Demonstrate the specific mutation
if 'col1' in columns_list:
    count = columns_list.count('col1')
    print(f"\n'col1' appears {count} times in the mutated list")
```

<details>

<summary>
Output showing columns list mutation
</summary>
```
Original columns list before call: ['col1', 'col2']
Expected error (file doesn't exist): 'NoneType' object has no attribute 'open'

After calling _read_orc:
Original copy:   ['col1', 'col2']
Modified list:   ['col1', 'col2', 'col1']
Lists are equal: False
Mutation occurred: True

'col1' appears 2 times in the mutated list
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Violation of Python Best Practices**: Functions should not mutate their input parameters unless that's their explicitly documented purpose. The Python community strongly discourages unexpected side effects.

2. **Inconsistent with Public API Design**: The public `read_orc` function (lines 97-98) creates a defensive copy of the columns list specifically to avoid mutation: `columns = [col for col in columns if col != index]`. This shows that non-mutation is the intended design pattern.

3. **Undocumented Side Effect**: Neither the function nor its parameters have any documentation indicating that the `columns` parameter will be mutated. Users calling this internal function would not expect their list to be modified.

4. **Potential for Subtle Bugs**: When the same columns list is reused after calling `_read_orc`, it will contain duplicate entries. In the example above, `'col1'` appears twice in the list after the call, which could lead to unexpected behavior in subsequent operations.

5. **The mutation only occurs under specific conditions** (when both `index` and `columns` are not None), making it harder to predict and debug.

## Relevant Context

- The bug occurs at line 113 in `/dask/dataframe/io/orc/core.py` where `columns.append(index)` directly modifies the input list.
- While `_read_orc` is an internal function (prefixed with underscore), it's still called through the public API via `dd.from_map()` at line 100.
- The public API currently protects users from this bug through defensive programming, but the internal function itself still exhibits problematic behavior.
- This could affect any code that directly calls `_read_orc` or if the public API is refactored in the future without maintaining the defensive copy.
- Documentation: [Dask ORC IO documentation](https://docs.dask.org/en/stable/dataframe-api.html#dask.dataframe.read_orc)
- Source code: [dask/dataframe/io/orc/core.py](https://github.com/dask/dask/blob/main/dask/dataframe/io/orc/core.py)

## Proposed Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,10 @@ def to_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        # Create a new list to avoid mutating the input parameter
+        columns = list(columns) + [index]
+    elif columns is not None:
+        # Ensure we're working with a copy
+        columns = list(columns)
     _df = engine.read_partition(
         fs,
         parts,
```