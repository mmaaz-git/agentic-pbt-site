import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from dask.dataframe.io.orc.core import _read_orc

with tempfile.TemporaryDirectory() as tmpdir:
    df = pd.DataFrame({'aa': [0], 'a': [0]})
    table = pa.Table.from_pandas(df)

    path = os.path.join(tmpdir, "test.orc")
    with open(path, 'wb') as f:
        orc.write_table(table, f)

    class FakeFS:
        def open(self, path, mode):
            return open(path, mode)

    fs = FakeFS()
    parts = [(path, None)]
    schema = {'aa': df['aa'].dtype, 'a': df['a'].dtype}

    columns = ['a']
    print(f"Before: columns = {columns}")

    _read_orc(
        parts,
        engine=ArrowORCEngine,
        fs=fs,
        schema=schema,
        index='aa',
        columns=columns
    )

    print(f"After: columns = {columns}")