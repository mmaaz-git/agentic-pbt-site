import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec

# Test the property-based test first
def test_columns_list_mutation():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.orc")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "idx": [10, 11, 12]})

        with open(file1, "wb") as f:
            orc.write_table(pa.Table.from_pandas(df), f)

        fs = fsspec.filesystem("file")
        schema = {"a": "int64", "b": "int64", "idx": "int64"}
        columns_original = ["a", "b"]
        columns_copy = columns_original.copy()

        _read_orc(
            parts=[(file1, None)],
            engine=ArrowORCEngine,
            fs=fs,
            schema=schema,
            index="idx",
            columns=columns_original,
        )

        assert columns_original == columns_copy, f"columns list was mutated: {columns_copy} -> {columns_original}"

print("Running property-based test...")
try:
    test_columns_list_mutation()
    print("Test passed (no mutation detected)")
except AssertionError as e:
    print(f"Test failed: {e}")

# Now run the reproduction example
print("\n" + "="*50)
print("Running reproduction example...")

tmpdir = tempfile.mkdtemp()
file1 = os.path.join(tmpdir, "file1.orc")
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "idx": [10, 11, 12]})

with open(file1, "wb") as f:
    orc.write_table(pa.Table.from_pandas(df), f)

fs = fsspec.filesystem("file")
schema = {"a": "int64", "b": "int64", "idx": "int64"}
columns = ["a", "b"]

print(f"Columns before: {columns}")

_read_orc(
    parts=[(file1, None)],
    engine=ArrowORCEngine,
    fs=fs,
    schema=schema,
    index="idx",
    columns=columns,
)

print(f"Columns after: {columns}")

# Clean up
import shutil
shutil.rmtree(tmpdir)