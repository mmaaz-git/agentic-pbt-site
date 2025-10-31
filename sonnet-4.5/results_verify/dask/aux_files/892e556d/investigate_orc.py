import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
import pyarrow.orc as orc
import os


def investigate_orc_files():
    """Investigate how the index is stored in ORC files"""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create a DataFrame with custom index
        pdf = pd.DataFrame({
            'int_col': [10, 20, 30, 40],
            'str_col': ['a', 'b', 'c', 'd']
        }, index=[100, 200, 300, 400])
        pdf.index.name = 'my_index'  # Give the index a name

        print("=== Original DataFrame ===")
        print(pdf)
        print(f"Index name: {pdf.index.name}")
        print(f"Index values: {list(pdf.index)}")

        # Write to ORC with Dask (multiple partitions)
        ddf = dd.from_pandas(pdf, npartitions=2)
        orc_path = f"{tmpdir}/test_orc"
        ddf.to_orc(orc_path, write_index=True)

        # List the ORC files created
        print(f"\n=== Files created in {orc_path} ===")
        for file in os.listdir(orc_path):
            print(f"  {file}")

        # Read each ORC file directly with PyArrow
        print("\n=== Examining ORC file contents ===")
        for i, file in enumerate(sorted(os.listdir(orc_path))):
            if file.endswith('.orc'):
                filepath = os.path.join(orc_path, file)
                print(f"\nPartition {i}: {file}")

                with open(filepath, 'rb') as f:
                    orc_file = orc.ORCFile(f)
                    # Read the schema
                    print(f"  Schema: {orc_file.schema}")

                    # Read the data
                    table = orc_file.read()
                    df = table.to_pandas()
                    print(f"  Data:\n{df}")
                    print(f"  Columns: {df.columns.tolist()}")
                    print(f"  Index: {df.index.tolist()}")

        # Now read back with Dask
        print("\n=== Reading back with Dask (no index specified) ===")
        result_ddf = dd.read_orc(orc_path)
        result = result_ddf.compute()
        print(result)
        print(f"Index values: {list(result.index)}")
        print(f"Index name: {result.index.name}")

        # Try reading with index column specified
        print("\n=== Reading back with Dask (index='my_index') ===")
        try:
            result_ddf2 = dd.read_orc(orc_path, index='my_index')
            result2 = result_ddf2.compute()
            print(result2)
            print(f"Index values: {list(result2.index)}")
            print(f"Index name: {result2.index.name}")
        except Exception as e:
            print(f"Error: {e}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_write_index_false():
    """Test what happens with write_index=False"""
    tmpdir = tempfile.mkdtemp()
    try:
        pdf = pd.DataFrame({
            'int_col': [10, 20, 30, 40],
        }, index=[100, 200, 300, 400])
        pdf.index.name = 'my_index'

        print("\n=== Test with write_index=False ===")
        print(f"Original DataFrame:\n{pdf}")

        ddf = dd.from_pandas(pdf, npartitions=2)
        orc_path = f"{tmpdir}/test_orc_no_index"
        ddf.to_orc(orc_path, write_index=False)

        # Read first ORC file directly
        files = sorted([f for f in os.listdir(orc_path) if f.endswith('.orc')])
        if files:
            filepath = os.path.join(orc_path, files[0])
            with open(filepath, 'rb') as f:
                orc_file = orc.ORCFile(f)
                print(f"\nORC Schema (write_index=False): {orc_file.schema}")
                table = orc_file.read()
                df = table.to_pandas()
                print(f"Data in first partition:\n{df}")

        result = dd.read_orc(orc_path).compute()
        print(f"\nResult DataFrame:\n{result}")
        print(f"Result index: {list(result.index)}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    investigate_orc_files()
    test_write_index_false()