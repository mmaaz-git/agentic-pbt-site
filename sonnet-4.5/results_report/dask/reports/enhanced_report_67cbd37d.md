# Bug Report: dask.dataframe.io.orc._read_orc Mutates Input Columns List

**Target**: `dask.dataframe.io.orc.core._read_orc`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_read_orc` function mutates its input `columns` list parameter by appending the `index` value to it, causing the list to grow incorrectly when processing multiple partitions and violating the principle that functions should not modify their input arguments.

## Property-Based Test

```python
from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.core import _read_orc

@given(st.lists(st.text(min_size=1), min_size=1), st.text(min_size=1))
@settings(max_examples=200)
def test_read_orc_does_not_mutate_columns(columns_list, index_name):
    original_columns = columns_list.copy()

    mock_engine = Mock()
    mock_engine.read_partition = Mock(return_value=Mock())

    mock_df = MagicMock()
    mock_df.set_index = Mock(return_value=MagicMock())
    mock_engine.read_partition.return_value = mock_df

    try:
        _read_orc(
            parts=[],
            engine=mock_engine,
            fs=Mock(),
            schema={},
            index=index_name,
            columns=columns_list
        )
    except Exception:
        pass

    assert columns_list == original_columns, f"columns list should not be mutated. Original: {original_columns}, After: {columns_list}"

if __name__ == "__main__":
    test_read_orc_does_not_mutate_columns()
```

<details>

<summary>
**Failing input**: `columns_list=['0'], index_name='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 32, in <module>
    test_read_orc_does_not_mutate_columns()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 6, in test_read_orc_does_not_mutate_columns
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 29, in test_read_orc_does_not_mutate_columns
    assert columns_list == original_columns, f"columns list should not be mutated. Original: {original_columns}, After: {columns_list}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: columns list should not be mutated. Original: ['0'], After: ['0', '0']
Falsifying example: test_read_orc_does_not_mutate_columns(
    # The test always failed when commented parts were varied together.
    columns_list=['0'],  # or any other generated value
    index_name='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from unittest.mock import Mock, MagicMock
from dask.dataframe.io.orc.core import _read_orc

# Test case demonstrating the mutation bug
columns_list = ['col1', 'col2']
index_name = 'idx'

print("Initial columns list:", columns_list)
print("Index name:", index_name)
print()

# Create mock objects to simulate the engine and filesystem
mock_engine = Mock()
mock_df = MagicMock()
mock_df.set_index = Mock(return_value=MagicMock())
mock_engine.read_partition.return_value = mock_df

# Simulate calling _read_orc multiple times (as would happen with multiple partitions)
print("Calling _read_orc for partition 1...")
_read_orc(
    parts=[],
    engine=mock_engine,
    fs=Mock(),
    schema={},
    index=index_name,
    columns=columns_list
)
print("After partition 1, columns list:", columns_list)

print("\nCalling _read_orc for partition 2...")
_read_orc(
    parts=[],
    engine=mock_engine,
    fs=Mock(),
    schema={},
    index=index_name,
    columns=columns_list
)
print("After partition 2, columns list:", columns_list)

print("\nCalling _read_orc for partition 3...")
_read_orc(
    parts=[],
    engine=mock_engine,
    fs=Mock(),
    schema={},
    index=index_name,
    columns=columns_list
)
print("After partition 3, columns list:", columns_list)

print("\n" + "="*50)
print("BUG CONFIRMED: The columns list was mutated!")
print("Expected: ['col1', 'col2']")
print("Actual:  ", columns_list)
print("The index 'idx' was appended", len(columns_list) - 2, "times")
```

<details>

<summary>
Columns list grows with repeated index appends
</summary>
```
Initial columns list: ['col1', 'col2']
Index name: idx

Calling _read_orc for partition 1...
After partition 1, columns list: ['col1', 'col2', 'idx']

Calling _read_orc for partition 2...
After partition 2, columns list: ['col1', 'col2', 'idx', 'idx']

Calling _read_orc for partition 3...
After partition 3, columns list: ['col1', 'col2', 'idx', 'idx', 'idx']

==================================================
BUG CONFIRMED: The columns list was mutated!
Expected: ['col1', 'col2']
Actual:   ['col1', 'col2', 'idx', 'idx', 'idx']
The index 'idx' was appended 3 times
```
</details>

## Why This Is A Bug

This bug violates the fundamental Python convention that functions should not mutate their input arguments unless that's their explicit purpose. The mutation occurs at line 113 in `dask/dataframe/io/orc/core.py` where `columns.append(index)` directly modifies the input list.

The bug manifests when `dd.from_map` calls `_read_orc` multiple times with the same `columns` list reference for different partitions. Each call appends the index again, causing the list to grow incorrectly with duplicate index entries. This can lead to:

1. **Incorrect column selection**: The engine receives a columns list with duplicate entries
2. **Memory inefficiency**: The list grows unnecessarily with each partition processed
3. **Unexpected behavior**: Callers who reuse the columns list will find it modified

While `_read_orc` is a private function (underscore prefix), it's still called by the public API through `dd.from_map` at line 99-108 of the same file, making this a real issue that affects end users when reading multi-partition ORC files.

## Relevant Context

The public `read_orc` function partially works around this issue at line 98 by creating a new list when removing the index from columns:
```python
if columns is not None and index in columns:
    columns = [col for col in columns if col != index]
```

However, this doesn't protect against all scenarios, particularly when the index is not already in the columns list. The mutation happens in the private `_read_orc` function which is called via `dd.from_map` for each partition.

Code location: `/dask/dataframe/io/orc/core.py:113`
Public API entry point: `dask.dataframe.read_orc()`

## Proposed Fix

```diff
--- a/dask/dataframe/io/orc/core.py
+++ b/dask/dataframe/io/orc/core.py
@@ -110,7 +110,7 @@ def _read_orc(parts, *, engine, fs, schema, index, columns=None):

 def _read_orc(parts, *, engine, fs, schema, index, columns=None):
     if index is not None and columns is not None:
-        columns.append(index)
+        columns = columns + [index]
     _df = engine.read_partition(
         fs,
         parts,
```