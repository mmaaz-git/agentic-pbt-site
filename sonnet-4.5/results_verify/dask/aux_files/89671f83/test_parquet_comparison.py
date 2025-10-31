import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    # Create a dataframe with a custom index
    df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    df.index = [100, 200]  # Set a custom index
    print("Original DataFrame:")
    print(df)
    print(f"Original index: {df.index.tolist()}")
    print()

    ddf = dd.from_pandas(df, npartitions=1)

    # Test Parquet round-trip
    parquet_path = f"{tmpdir}/test_parquet"
    ddf.to_parquet(parquet_path, write_index=True)

    result_parquet = dd.read_parquet(parquet_path)
    result_parquet_df = result_parquet.compute()

    print("Parquet Round-trip Result:")
    print(result_parquet_df)
    print(f"Parquet index: {result_parquet_df.index.tolist()}")
    print(f"Parquet columns: {result_parquet_df.columns.tolist()}")
    print()

    # Test ORC round-trip
    orc_path = f"{tmpdir}/test_orc"
    ddf.to_orc(orc_path, write_index=True)

    result_orc = dd.read_orc(orc_path)
    result_orc_df = result_orc.compute()

    print("ORC Round-trip Result (without index param):")
    print(result_orc_df)
    print(f"ORC index: {result_orc_df.index.tolist()}")
    print(f"ORC columns: {result_orc_df.columns.tolist()}")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)