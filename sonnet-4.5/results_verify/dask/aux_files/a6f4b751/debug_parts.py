import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from fsspec.core import get_fs_token_paths

print("Debugging what parts look like for empty ORC files")
print("=" * 60)

tmpdir = tempfile.mkdtemp()
try:
    # Create an empty DataFrame
    df_pandas = pd.DataFrame({'col_a': [], 'col_b': []})
    df = dd.from_pandas(df_pandas, npartitions=1)
    dd.to_orc(df, tmpdir, write_index=False)

    # Get engine
    engine = ArrowORCEngine

    # Process file path(s)
    storage_options = {}
    fs, fs_token, paths = get_fs_token_paths(
        tmpdir, mode="rb", storage_options=storage_options
    )

    print(f"File paths found: {paths}")

    # Let backend engine generate a list of parts
    parts, schema, meta = engine.read_metadata(
        fs,
        paths,
        columns=None,
        index=None,
        split_stripes=1,
        aggregate_files=None,
    )

    print(f"\nparts: {parts}")
    print(f"  Type: {type(parts)}")
    print(f"  Length: {len(parts)}")
    print(f"\nschema: {schema}")
    print(f"\nmeta (DataFrame preview):")
    print(meta)
    print(f"  Shape: {meta.shape}")
    print(f"  Columns: {list(meta.columns)}")

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)