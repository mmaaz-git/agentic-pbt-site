import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.core import _read_orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec

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