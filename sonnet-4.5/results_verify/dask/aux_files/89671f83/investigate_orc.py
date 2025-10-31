import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
import pyarrow.orc as orc
import pyarrow as pa

tmpdir = tempfile.mkdtemp()
try:
    # Create a dataframe with a meaningful index
    df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    df.index = [100, 200]  # Set a custom index
    print("Original DataFrame:")
    print(df)
    print(f"Original index: {df.index.tolist()}")
    print(f"Original index name: {df.index.name}")
    print()

    ddf = dd.from_pandas(df, npartitions=1)
    path = f"{tmpdir}/test_orc"
    ddf.to_orc(path, write_index=True)

    # Let's inspect what was actually written to the ORC file
    import os
    orc_files = [f for f in os.listdir(path) if f.endswith('.orc')]
    print(f"ORC files created: {orc_files}")

    if orc_files:
        orc_path = os.path.join(path, orc_files[0])
        # Read the ORC file directly with pyarrow
        orc_file = orc.ORCFile(orc_path)
        print(f"\nORC file schema: {orc_file.schema}")
        print(f"ORC file num rows: {orc_file.nrows}")

        # Read as pyarrow table
        table = orc_file.read()
        print(f"\nPyArrow table columns: {table.column_names}")
        print(f"PyArrow table:\n{table.to_pandas()}")

    print("\n" + "="*50)
    print("Now reading back with dask.read_orc():")
    result = dd.read_orc(path)
    result_df = result.compute()

    print("\nResult DataFrame:")
    print(result_df)
    print(f"Result index: {result_df.index.tolist()}")
    print(f"Result columns: {result_df.columns.tolist()}")

    print("\n" + "="*50)
    print("Reading with index specified:")
    result2 = dd.read_orc(path, index='__index_level_0__')
    result2_df = result2.compute()
    print("\nResult2 DataFrame:")
    print(result2_df)
    print(f"Result2 index: {result2_df.index.tolist()}")
    print(f"Result2 columns: {result2_df.columns.tolist()}")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)