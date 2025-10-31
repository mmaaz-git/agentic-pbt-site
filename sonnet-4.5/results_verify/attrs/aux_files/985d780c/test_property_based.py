import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import tempfile
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from hypothesis import given, strategies as st, settings, assume
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from dask.dataframe.io.orc.core import _read_orc


@settings(max_examples=100)
@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=20),
    st.lists(st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)), min_size=1, max_size=5).map(lambda x: list(set(x)))
)
def test_columns_not_mutated_in_read_orc(data, columns):
    assume(len(columns) >= 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        df_data = {col: data for col in columns}
        df = pd.DataFrame(df_data)
        table = pa.Table.from_pandas(df)

        path = os.path.join(tmpdir, "test.orc")
        with open(path, 'wb') as f:
            orc.write_table(table, f)

        class FakeFS:
            def open(self, path, mode):
                return open(path, mode)

        fs = FakeFS()
        index_col = columns[0]
        read_columns = columns[1:].copy()
        original_columns = read_columns.copy()

        parts = [(path, None)]
        schema = {col: df[col].dtype for col in columns}

        _read_orc(
            parts,
            engine=ArrowORCEngine,
            fs=fs,
            schema=schema,
            index=index_col,
            columns=read_columns
        )

        assert read_columns == original_columns, f"Columns were mutated! Before: {original_columns}, After: {read_columns}"


if __name__ == "__main__":
    test_columns_not_mutated_in_read_orc()
    print("Test completed")