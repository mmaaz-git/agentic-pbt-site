import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

print("Testing bug report: Empty DataFrame ORC round-trip")
print("=" * 60)

tmpdir = tempfile.mkdtemp()
try:
    # Create an empty DataFrame
    df_pandas = pd.DataFrame({'col_a': [], 'col_b': []})
    print(f"Created empty pandas DataFrame with columns: {list(df_pandas.columns)}")
    print(f"DataFrame shape: {df_pandas.shape}")

    # Convert to Dask DataFrame
    df = dd.from_pandas(df_pandas, npartitions=1)
    print("Converted to Dask DataFrame")

    # Write to ORC
    dd.to_orc(df, tmpdir, write_index=False)
    print(f"Successfully wrote empty DataFrame to ORC at: {tmpdir}")

    # Try to read back
    print("Attempting to read ORC file...")
    df_read = dd.read_orc(tmpdir)
    print("Successfully created Dask DataFrame from ORC")

    # Try to compute
    df_result = df_read.compute()
    print(f"Successfully computed result: shape={df_result.shape}, columns={list(df_result.columns)}")

except Exception as e:
    print(f"\nERROR occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nCleaned up temporary directory: {tmpdir}")