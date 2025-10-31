import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
import fsspec


def test_schema_mismatch_not_detected_when_split_stripes_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.orc")
        file2 = os.path.join(tmpdir, "file2.orc")

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        table1 = pa.Table.from_pandas(df1)

        df2 = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})
        table2 = pa.Table.from_pandas(df2)

        with open(file1, "wb") as f:
            orc.write_table(table1, f)
        with open(file2, "wb") as f:
            orc.write_table(table2, f)

        fs = fsspec.filesystem("file")

        parts, schema, meta = ArrowORCEngine.read_metadata(
            fs=fs,
            paths=[file1, file2],
            columns=None,
            index=None,
            split_stripes=False,
            aggregate_files=False,
        )
        assert False, "Should have raised ValueError for incompatible schemas"


if __name__ == "__main__":
    try:
        test_schema_mismatch_not_detected_when_split_stripes_false()
        print("Test PASSED - but it shouldn't have!")
    except ValueError as e:
        if "Incompatible schemas" in str(e):
            print(f"Test FAILED as expected - caught schema mismatch: {e}")
        else:
            raise
    except AssertionError as e:
        print(f"Test FAILED - {e}")