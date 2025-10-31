import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd

# First, let's test the simple reproduction case
print("Testing simple reproduction case...")
tmpdir = tempfile.mkdtemp()
try:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    ddf = dd.from_pandas(df, npartitions=1)

    orc_path = f"{tmpdir}/test.orc"
    ddf.to_orc(orc_path)

    # This should fail according to the bug report
    print("Attempting to read ORC with columns=['a'] and index='c'...")
    try:
        result = dd.read_orc(orc_path, columns=["a"], index="c")
        computed = result.compute()
        print(f"Success! Result:\n{computed}")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")

    # Let's also test the working case mentioned
    print("\nTesting the working case with columns=['a', 'c'] and index='c'...")
    try:
        result = dd.read_orc(orc_path, columns=["a", "c"], index="c")
        computed = result.compute()
        print(f"Success! Result:\n{computed}")
        print(f"Columns: {list(computed.columns)}")
        print(f"Index name: {computed.index.name}")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)