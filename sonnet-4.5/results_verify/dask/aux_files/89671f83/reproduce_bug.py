import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    df = pd.DataFrame({"a": [0, 0], "b": [0, 0]})
    print("Original DataFrame:")
    print(df)
    print(f"Original index: {df.index.tolist()}")

    ddf = dd.from_pandas(df, npartitions=2)
    path = f"{tmpdir}/test_orc"
    ddf.to_orc(path, write_index=True)

    result = dd.read_orc(path)
    result_df = result.compute()

    print("\nResult DataFrame:")
    print(result_df)
    print(f"Result index: {result_df.index.tolist()}")
    print(f"Result columns: {result_df.columns.tolist()}")

    print("\nExpected index: [0, 1]")
    print(f"Actual index: {result_df.index.tolist()}")
    print("Index preserved:", result_df.index.tolist() == [0, 1])
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)