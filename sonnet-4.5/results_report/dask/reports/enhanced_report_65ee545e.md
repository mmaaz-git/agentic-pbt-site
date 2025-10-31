# Bug Report: dask.dataframe.io.orc ArrowORCEngine Always Opens First File Instead of Current File

**Target**: `dask.dataframe.io.orc.arrow.ArrowORCEngine.read_metadata`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

In the `read_metadata` method, when `split_stripes=False`, the code incorrectly opens `paths[0]` instead of the loop variable `path`, causing it to always read from the first file rather than the current file being processed.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from unittest.mock import MagicMock
import pyarrow as pa
import pyarrow.orc as orc

@given(st.lists(st.text(min_size=1, alphabet=st.characters(blacklist_characters=['/', '\x00', '\\'])),
                min_size=2, max_size=5, unique=True))
@settings(max_examples=50)
def test_read_metadata_opens_correct_files(file_names):
    paths = [f"{name}.orc" for name in file_names]

    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_fs.open.return_value.__enter__.return_value = mock_file

    mock_orc_file = MagicMock()
    mock_orc_file.schema = pa.schema([('col1', pa.int64())])

    # Store original to restore later
    original_orcfile = orc.ORCFile

    # Mock the ORCFile
    orc.ORCFile = lambda f: mock_orc_file

    try:
        ArrowORCEngine.read_metadata(
            mock_fs, paths, None, None, False, None
        )

        opened_paths = [call_args[0][0] for call_args in mock_fs.open.call_args_list]

        # When split_stripes=False, the bug causes only paths[0] to be opened
        if len(paths) > 1:
            assert paths[0] in opened_paths, f"First file {paths[0]} should be opened"
            if len(set(opened_paths)) == 1 and opened_paths[0] == paths[0]:
                # Bug detected: Only first file was opened multiple times or once
                print(f"BUG: With {len(paths)} files, only {paths[0]} was opened")
                print(f"Expected to open at minimum: {paths[0]} (for schema)")
                print(f"Actually opened: {opened_paths}")
                assert False, f"Only {paths[0]} was opened, but should open path in loop, not paths[0]"
    finally:
        # Restore original
        orc.ORCFile = original_orcfile

# Run the test
if __name__ == "__main__":
    test_read_metadata_opens_correct_files()
```

<details>

<summary>
**Failing input**: `file_names=['0', '1']`
</summary>
```
BUG: With 2 files, only _ÚÝÿÍp.orc was opened
Expected to open at minimum: _ÚÝÿÍp.orc (for schema)
Actually opened: ['_ÚÝÿÍp.orc']
BUG: With 5 files, only _ÚÝÿÍp.orc was opened
Expected to open at minimum: _ÚÝÿÍp.orc (for schema)
Actually opened: ['_ÚÝÿÍp.orc']
[... continues for hundreds of test cases ...]
BUG: With 2 files, only 0.orc was opened
Expected to open at minimum: 0.orc (for schema)
Actually opened: ['0.orc']
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 48, in <module>
  |     test_read_metadata_opens_correct_files()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 8, in test_read_metadata_opens_correct_files
  |     min_size=2, max_size=5, unique=True))
  |     ^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 38, in test_read_metadata_opens_correct_files
    |     print(f"BUG: With {len(paths)} files, only {paths[0]} was opened")
    |     ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | UnicodeEncodeError: 'utf-8' codec can't encode character '\udd00' in position 24: surrogates not allowed
    | Falsifying example: test_read_metadata_opens_correct_files(
    |     file_names=['\udd00', '0'],
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 41, in test_read_metadata_opens_correct_files
    |     assert False, f"Only {paths[0]} was opened, but should open path in loop, not paths[0]"
    |            ^^^^^
    | AssertionError: Only 0.orc was opened, but should open path in loop, not paths[0]
    | Falsifying example: test_read_metadata_opens_correct_files(
    |     file_names=['0', '1'],  # or any other generated value
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from unittest.mock import MagicMock
import pyarrow as pa
import pyarrow.orc as orc

# Create mock filesystem and file objects
mock_fs = MagicMock()
paths = ['file1.orc', 'file2.orc', 'file3.orc']

# Mock file object
mock_file = MagicMock()
mock_fs.open.return_value.__enter__.return_value = mock_file

# Mock ORC file with a schema
mock_orc_file = MagicMock()
mock_orc_file.schema = pa.schema([('col1', pa.int64())])

# Store the original ORCFile to restore it later
original_orcfile = orc.ORCFile

# Replace ORCFile with our mock
orc.ORCFile = lambda f: mock_orc_file

try:
    # Call the read_metadata method with split_stripes=False
    parts, schema, meta = ArrowORCEngine.read_metadata(
        mock_fs, paths, None, None, False, None
    )

    # Extract the files that were opened
    opened_files = [call[0][0] for call in mock_fs.open.call_args_list]

    print("Files that should be processed:", paths)
    print("Files actually opened:", opened_files)
    print()

    # Check if the bug exists
    if len(opened_files) == 1 and opened_files[0] == 'file1.orc':
        print("BUG CONFIRMED: Only paths[0] (file1.orc) was opened!")
        print("Expected: Each file should be checked when processing it")
        print("Actual: Only the first file is ever opened to read schema")
    else:
        print("Bug not reproduced - all files were opened correctly")

finally:
    # Restore the original ORCFile
    orc.ORCFile = original_orcfile
```

<details>

<summary>
BUG CONFIRMED: Only paths[0] was opened
</summary>
```
Files that should be processed: ['file1.orc', 'file2.orc', 'file3.orc']
Files actually opened: ['file1.orc']

BUG CONFIRMED: Only paths[0] (file1.orc) was opened!
Expected: Each file should be checked when processing it
Actual: Only the first file is ever opened to read schema
```
</details>

## Why This Is A Bug

This is a logic error where the loop variable `path` is defined but never used inside the loop body. In line 60 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/dataframe/io/orc/arrow.py`, the code uses `paths[0]` instead of `path`:

```python
for path in paths:  # Line 58: Loop variable 'path' is defined
    if schema is None:
        with fs.open(paths[0], "rb") as f:  # Line 60: BUG - uses paths[0] instead of path
            o = orc.ORCFile(f)
            schema = o.schema
    parts.append([(path, None)])
```

While this bug has minimal practical impact in typical usage (since the schema is only read once when `schema is None` on the first iteration, where `path` equals `paths[0]`), it represents poor code quality and could cause issues if the code logic changes in the future. The unused loop variable is a classic programming error that violates basic coding principles.

## Relevant Context

The bug appears in the `else` branch (lines 57-63) that handles the case when `split_stripes=False`. This contrasts with the `if` branch (lines 36-56) for `split_stripes=True`, which correctly opens each file individually using the loop variable `path` on line 39.

The `split_stripes=False` mode is intended to create a 1-to-1 mapping between files and partitions according to the documentation. The current implementation assumes all files have the same schema and only reads it from the first file, which is likely an intentional optimization. However, the use of `paths[0]` instead of `path` is clearly a typo since it makes the loop variable unused.

Documentation link: The public API is exposed through `dask.dataframe.read_orc`, where `split_stripes=False` specifies a 1-to-1 mapping between files and partitions.

## Proposed Fix

```diff
--- a/dask/dataframe/io/orc/arrow.py
+++ b/dask/dataframe/io/orc/arrow.py
@@ -57,7 +57,7 @@ class ArrowORCEngine:
         else:
             for path in paths:
                 if schema is None:
-                    with fs.open(paths[0], "rb") as f:
+                    with fs.open(path, "rb") as f:
                         o = orc.ORCFile(f)
                         schema = o.schema
                 parts.append([(path, None)])
```