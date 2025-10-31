"""Test if the mutation could affect multiple partition reads"""

import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec

print("Testing if mutation could affect multiple partition reads...")
print()

# Create multiple ORC files
with tempfile.TemporaryDirectory() as tmpdir:
    file_paths = []
    for i in range(3):
        file_path = os.path.join(tmpdir, f"test_{i}.orc")
        df = pd.DataFrame({
            "a": [i*10 + 1, i*10 + 2, i*10 + 3],
            "b": [i*10 + 4, i*10 + 5, i*10 + 6],
            "idx": [i*10 + 10, i*10 + 11, i*10 + 12]
        })
        table = pa.Table.from_pandas(df)
        with open(file_path, "wb") as f:
            orc.write_table(table, f)
        file_paths.append((file_path, None))

    fs = fsspec.filesystem("file")
    schema = {"a": "int64", "b": "int64", "idx": "int64"}

    # Simulate what dd.from_map might do - reuse the same columns list
    columns_shared = ["a", "b"]
    print(f"Initial columns list: {columns_shared}")
    print()

    for i, part in enumerate(file_paths):
        print(f"Processing partition {i}...")
        print(f"  Before call: columns = {columns_shared}")

        result = _read_orc(
            parts=[part],
            engine=ArrowORCEngine,
            fs=fs,
            schema=schema,
            index="idx",
            columns=columns_shared,  # Same list object passed each time!
        )

        print(f"  After call:  columns = {columns_shared}")

    print()
    print(f"Final columns list: {columns_shared}")
    print(f"Expected: ['a', 'b']")
    print(f"Got multiple 'idx' appended? {columns_shared.count('idx')} times")

    if columns_shared != ["a", "b"]:
        print()
        print("⚠️  BUG IMPACT: The mutation accumulates across multiple calls!")
        print("This could cause unexpected behavior when processing multiple partitions.")