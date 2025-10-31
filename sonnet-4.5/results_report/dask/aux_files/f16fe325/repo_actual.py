import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

tmpdir = tempfile.mkdtemp()
try:
    # Create an empty pandas DataFrame with columns but no rows
    df_pandas = pd.DataFrame({'col_a': [], 'col_b': []})
    print(f"Created empty DataFrame: shape={df_pandas.shape}, columns={list(df_pandas.columns)}")

    # Convert to Dask DataFrame
    df = dd.from_pandas(df_pandas, npartitions=1)
    print(f"Converted to Dask DataFrame: npartitions={df.npartitions}")

    # Write to ORC format
    dd.to_orc(df, tmpdir, write_index=False)
    print(f"Successfully wrote empty DataFrame to ORC in {tmpdir}")

    # List the ORC files created
    import os
    files = os.listdir(tmpdir)
    print(f"ORC files created: {files}")

    # Try to read the ORC file back
    print("Attempting to read ORC file...")
    df_read = dd.read_orc(tmpdir)
    print("Successfully read ORC file")

    df_result = df_read.compute()
    print(f"Result DataFrame: shape={df_result.shape}, columns={list(df_result.columns)}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
