# Bug Report: dask.dataframe.io.orc._read_orc Mutates Input Columns List

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates its input `columns` list parameter by appending the `index` value to it, violating the Python convention that functions should not modify their input parameters unless explicitly documented.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for dask ORC columns mutation bug"""

from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine


@given(
    index=st.one_of(st.none(), st.text(min_size=1)),
    columns=st.one_of(st.none(), st.lists(st.text(min_size=1), min_size=1, max_size=5))
)
@settings(max_examples=100)
def test_read_orc_does_not_mutate_columns(index, columns):
    """Test that _read_orc does not mutate the input columns list"""
    if columns is not None:
        columns_before = list(columns)

        try:
            _read_orc(
                parts=[],
                engine=ArrowORCEngine,
                fs=None,
                schema={},
                index=index,
                columns=columns
            )
        except:
            pass

        assert columns == columns_before, f"columns was mutated from {columns_before} to {columns}"


if __name__ == "__main__":
    # Run the test
    print("Running property-based test for _read_orc columns mutation...")
    test_read_orc_does_not_mutate_columns()
```

<details>

<summary>
**Failing input**: `index='0', columns=['0']`
</summary>
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from hypo import test_read_orc_does_not_mutate_columns; test_read_orc_does_not_mutate_columns()
                                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 10, in test_read_orc_does_not_mutate_columns
    index=st.one_of(st.none(), st.text(min_size=1)),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 31, in test_read_orc_does_not_mutate_columns
    assert columns == columns_before, f"columns was mutated from {columns_before} to {columns}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: columns was mutated from ['0'] to ['0', '0']
Falsifying example: test_read_orc_does_not_mutate_columns(
    index='0',
    columns=['0'],
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstration of the dask ORC columns mutation bug"""

from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine

# Test case 1: Basic mutation demonstration
print("=== Test Case 1: Basic Mutation ===")
columns = ['col1', 'col2']
print(f"Before calling _read_orc: columns = {columns}")
print(f"Object ID before: {id(columns)}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index='col1',
        columns=columns
    )
except Exception as e:
    pass  # Expected to fail due to empty parts

print(f"After calling _read_orc: columns = {columns}")
print(f"Object ID after: {id(columns)}")
print(f"Mutation occurred: {columns == ['col1', 'col2', 'col1']}")
print()

# Test case 2: The specific failing example from Hypothesis
print("=== Test Case 2: Hypothesis Failing Input ===")
columns = ['0']
print(f"Before calling _read_orc: columns = {columns}")

try:
    _read_orc(
        parts=[],
        engine=ArrowORCEngine,
        fs=None,
        schema={},
        index='0',
        columns=columns
    )
except Exception as e:
    pass

print(f"After calling _read_orc: columns = {columns}")
print(f"Expected: ['0'], Got: {columns}")
print()

# Test case 3: Multiple calls accumulate mutations
print("=== Test Case 3: Accumulation of Mutations ===")
columns = ['col1', 'col2']
print(f"Initial columns: {columns}")

for i in range(3):
    try:
        _read_orc(
            parts=[],
            engine=ArrowORCEngine,
            fs=None,
            schema={},
            index=f'idx{i}',
            columns=columns
        )
    except Exception as e:
        pass
    print(f"After call {i+1}: columns = {columns}")

print()
print("=== Summary ===")
print("The _read_orc function mutates its input columns list by appending")
print("the index parameter to it. This violates Python's principle that")
print("functions should not mutate their inputs unless explicitly documented.")
```

<details>

<summary>
Demonstration of in-place mutation and accumulation effects
</summary>
```
=== Test Case 1: Basic Mutation ===
Before calling _read_orc: columns = ['col1', 'col2']
Object ID before: 136553542011712
After calling _read_orc: columns = ['col1', 'col2', 'col1']
Object ID after: 136553542011712
Mutation occurred: True

=== Test Case 2: Hypothesis Failing Input ===
Before calling _read_orc: columns = ['0']
After calling _read_orc: columns = ['0', '0']
Expected: ['0'], Got: ['0', '0']

=== Test Case 3: Accumulation of Mutations ===
Initial columns: ['col1', 'col2']
After call 1: columns = ['col1', 'col2', 'idx0']
After call 2: columns = ['col1', 'col2', 'idx0', 'idx1']
After call 3: columns = ['col1', 'col2', 'idx0', 'idx1', 'idx2']

=== Summary ===
The _read_orc function mutates its input columns list by appending
the index parameter to it. This violates Python's principle that
functions should not mutate their inputs unless explicitly documented.
```
</details>

## Why This Is A Bug

The `_read_orc` function at line 113 in `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/io/orc/core.py` directly modifies the input `columns` list using `columns.append(index)`. This violates several important principles:

1. **Python convention violation**: Functions should not mutate their input parameters unless explicitly documented. The function has no docstring indicating this side effect.

2. **Inconsistency with public API**: The public `read_orc` function (line 98) already protects against this issue by creating a new list: `columns = [col for col in columns if col != index]`. This shows the developers understand that mutation should be avoided, making the mutation in `_read_orc` inconsistent.

3. **Accumulation problem**: Multiple calls with the same list cause accumulation of index values, as demonstrated in Test Case 3 where the list grows from `['col1', 'col2']` to `['col1', 'col2', 'idx0', 'idx1', 'idx2']`.

4. **Same object mutation**: The mutation happens in-place (same object ID), meaning any code holding a reference to the original list sees unexpected changes.

## Relevant Context

The `_read_orc` function is a private internal function (indicated by the leading underscore) called by the public `read_orc` function via `dd.from_map` at line 99-108. While the public API protects against this issue in most cases, the bug still exists and could affect:

- Direct calls to the private `_read_orc` function (discouraged but possible)
- Future refactorings that change how `read_orc` handles the columns list
- Other internal code paths that might use `_read_orc`

The bug is located in the dask library's ORC I/O module, specifically in the core implementation that handles reading ORC files into Dask DataFrames.

## Proposed Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,8 @@ def to_orc(

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = list(columns)  # Create a copy to avoid mutation
+        columns.append(index)
     _df = engine.read_partition(
         fs,
         parts,
```