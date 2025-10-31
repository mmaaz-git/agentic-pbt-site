import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec

tmpdir = tempfile.mkdtemp()
file1 = os.path.join(tmpdir, "file1.orc")
file2 = os.path.join(tmpdir, "file2.orc")

df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df2 = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})

with open(file1, "wb") as f:
    orc.write_table(pa.Table.from_pandas(df1), f)
with open(file2, "wb") as f:
    orc.write_table(pa.Table.from_pandas(df2), f)

fs = fsspec.filesystem("file")
parts, schema, meta = ArrowORCEngine.read_metadata(
    fs=fs,
    paths=[file1, file2],
    columns=None,
    index=None,
    split_stripes=False,
    aggregate_files=False,
)

print(f"Schema detected: {schema}")
print("No error raised - schema mismatch was not detected!")