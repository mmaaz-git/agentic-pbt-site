# Bug Report: dask.dataframe.io.orc ArrowORCEngine Opens Wrong File

**Target**: `dask.dataframe.io.orc.arrow.ArrowORCEngine.read_metadata`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `split_stripes=False`, the `read_metadata` method opens `paths[0]` instead of the current `path` in the loop, causing it to always read the schema from the first file rather than the current file being processed.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from unittest.mock import MagicMock, mock_open, call
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
    mock_orc_file.schema = MagicMock()

    with mock.patch('pyarrow.orc.ORCFile', return_value=mock_orc_file):
        try:
            ArrowORCEngine.read_metadata(
                mock_fs, paths, None, None, False, None
            )
        except:
            pass

    opened_paths = [call_args[0][0] for call_args in mock_fs.open.call_args_list]

    if len(paths) > 1:
        assert paths[0] in opened_paths, f"First file {paths[0]} should be opened"
        if len(set(opened_paths)) == 1:
            assert False, f"Only {paths[0]} was opened, but should open all files when schema is None initially"
```

**Failing input**: Any list with multiple file paths when `split_stripes=False`

## Reproducing the Bug

```python
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from unittest.mock import MagicMock, mock_open
import pyarrow as pa

mock_fs = MagicMock()
paths = ['file1.orc', 'file2.orc', 'file3.orc']

mock_file = MagicMock()
mock_fs.open.return_value.__enter__.return_value = mock_file

mock_orc_file = MagicMock()
mock_orc_file.schema = pa.schema([('col1', pa.int64())])

import pyarrow.orc as orc
original_orcfile = orc.ORCFile
orc.ORCFile = lambda f: mock_orc_file

try:
    ArrowORCEngine.read_metadata(
        mock_fs, paths, None, None, False, None
    )

    opened_files = [call[0][0] for call in mock_fs.open.call_args_list]
    print(f"Files opened: {opened_files}")
    print(f"Expected to open: {paths}")

    if 'file2.orc' not in opened_files or 'file3.orc' not in opened_files:
        print("BUG: Not all files were opened! Only paths[0] was opened for schema.")
finally:
    orc.ORCFile = original_orcfile
```

## Why This Is A Bug

In the `else` block (when `split_stripes=False`) at line 58-63 of `arrow.py`, the code iterates over `path in paths` but then opens `paths[0]` on line 60:

```python
for path in paths:
    if schema is None:
        with fs.open(paths[0], "rb") as f:  # BUG: Should be 'path'
            o = orc.ORCFile(f)
            schema = o.schema
    parts.append([(path, None)])
```

This means:
1. It always opens the first file (`paths[0]`) to get the schema, regardless of which `path` is being processed
2. If files have incompatible schemas, the validation that should happen (like on line 43 in the `split_stripes=True` branch) doesn't occur properly
3. The loop variable `path` is unused for opening the file, which is clearly a logic error

## Fix

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