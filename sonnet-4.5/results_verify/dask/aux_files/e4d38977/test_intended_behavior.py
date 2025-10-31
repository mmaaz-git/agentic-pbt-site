#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

print("=" * 60)
print("Analyzing Intended Behavior")
print("=" * 60)

print("\nLooking at the code structure in arrow.py:")
print("\nWhen split_stripes=True (lines 36-56):")
print("  for path in paths:")
print("    with fs.open(path, 'rb') as f:  # Opens each file correctly")
print("      o = orc.ORCFile(f)")
print("      if schema is None:")
print("        schema = o.schema")
print("      elif schema != o.schema:")
print("        raise ValueError('Incompatible schemas...')")
print("\nWhen split_stripes=False (lines 58-63):")
print("  for path in paths:")
print("    if schema is None:")
print("      with fs.open(paths[0], 'rb') as f:  # BUG: Always opens first file")
print("        o = orc.ORCFile(f)")
print("        schema = o.schema")
print("    parts.append([(path, None)])")

print("\n" + "=" * 60)
print("What the Developer Likely Intended")
print("=" * 60)

print("\nLikely intention for split_stripes=False:")
print("1. Only read the schema from the FIRST file encountered")
print("2. Assume all other files have the same schema (optimization)")
print("3. Don't validate schemas across files (for performance)")

print("\nBut the implementation is wrong because:")
print("- The condition 'if schema is None' only triggers on first iteration")
print("- After that, schema is set, so no file opening happens")
print("- So using paths[0] vs path makes no functional difference...")
print("- EXCEPT it's clearly a typo because 'path' is the loop variable!")

print("\n" + "=" * 60)
print("Testing What Happens After Fixing the Bug")
print("=" * 60)

from dask.dataframe.io.orc.arrow import ArrowORCEngine
from unittest.mock import MagicMock
import pyarrow as pa
import pyarrow.orc as orc

mock_fs = MagicMock()
paths = ['file1.orc', 'file2.orc', 'file3.orc']

opened_files = []

def mock_open_side_effect(path, mode):
    opened_files.append(path)
    mock_file = MagicMock()
    mock_file.__enter__ = lambda self: self
    mock_file.__exit__ = lambda self, *args: None
    return mock_file

mock_fs.open.side_effect = mock_open_side_effect

mock_orc_file = MagicMock()
mock_orc_file.schema = pa.schema([('col1', pa.int64())])

original_orcfile = orc.ORCFile
orc.ORCFile = lambda f: mock_orc_file

try:
    result = ArrowORCEngine.read_metadata(
        mock_fs, paths, None, None, split_stripes=False, aggregate_files=None
    )

    print(f"Files opened with current bug: {opened_files}")
    print(f"Result: Opens '{opened_files[0]}' (always first file due to bug)")

    print("\nIf bug was fixed (paths[0] -> path):")
    print(f"Would open: '{paths[0]}' (first file in loop when schema is None)")
    print("Then schema would be set, so no more files would be opened")
    print("\nSo fixing the bug changes:")
    print("  FROM: Always opening paths[0] regardless of loop iteration")
    print("  TO:   Opening the current 'path' when schema is None")
    print("\nIn practice, both result in opening the first file once,")
    print("BUT the bug is still wrong because:")
    print("1. It's clearly a typo (unused loop variable)")
    print("2. It's inconsistent with the split_stripes=True branch")
    print("3. It could cause issues if the code is modified in the future")

finally:
    orc.ORCFile = original_orcfile

print("\n" + "=" * 60)
print("Edge Case Analysis")
print("=" * 60)

print("\nWhat if paths list is reordered or filtered before this loop?")
print("Current bug: Would still open paths[0]")
print("Fixed code: Would open the first 'path' in iteration order")
print("\nWhat if someone modifies the code to reset schema to None mid-loop?")
print("Current bug: Would keep opening paths[0]")
print("Fixed code: Would open the current path being processed")
print("\nConclusion: The bug is real and should be fixed!")