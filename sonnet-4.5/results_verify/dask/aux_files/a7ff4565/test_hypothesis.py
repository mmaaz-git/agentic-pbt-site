import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import pandas as pd
import dask.dataframe as dd
from hypothesis import given, settings
import hypothesis.extra.pandas as pd_st


@given(pd_st.data_frames([
    pd_st.column('a', dtype=int),
    pd_st.column('b', dtype=float),
    pd_st.column('c', dtype=str),
]))
@settings(max_examples=100)
def test_from_pandas_round_trip(df):
    ddf = dd.from_pandas(df, npartitions=2)
    result = ddf.compute()

    pd.testing.assert_frame_equal(result, df, check_dtype=True)

if __name__ == "__main__":
    test_from_pandas_round_trip()