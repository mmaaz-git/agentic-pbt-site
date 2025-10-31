import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
import pyarrow.orc as orc
import pyarrow as pa
import os

print("Testing PyArrow's ability to handle empty ORC files")
print("=" * 60)

tmpdir = tempfile.mkdtemp()
try:
    # Create an empty DataFrame
    df_pandas = pd.DataFrame({'col_a': [], 'col_b': []})
    print(f"Created empty pandas DataFrame: shape={df_pandas.shape}, columns={list(df_pandas.columns)}")

    # Convert to Dask DataFrame and write to ORC
    df = dd.from_pandas(df_pandas, npartitions=1)
    dd.to_orc(df, tmpdir, write_index=False)

    # List files in the directory
    files = os.listdir(tmpdir)
    print(f"\nFiles written to {tmpdir}: {files}")

    # Try to read with PyArrow directly
    for file in files:
        if file.endswith('.orc'):
            filepath = os.path.join(tmpdir, file)
            print(f"\nReading {file} with PyArrow...")

            # Read using PyArrow ORC
            orc_file = orc.ORCFile(filepath)
            print(f"  Schema: {orc_file.schema}")
            print(f"  Number of stripes: {orc_file.nstripes}")
            print(f"  Number of rows: {orc_file.nrows}")

            # Read the table
            table = orc_file.read()
            print(f"  Table shape: {table.shape}")
            print(f"  Table columns: {table.column_names}")

            # Convert to pandas
            df_result = table.to_pandas()
            print(f"  Pandas DataFrame shape: {df_result.shape}")
            print(f"  Pandas DataFrame columns: {list(df_result.columns)}")

    print("\nConclusion: PyArrow can successfully read empty ORC files!")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)