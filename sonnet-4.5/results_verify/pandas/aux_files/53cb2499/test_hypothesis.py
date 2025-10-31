import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings


@given(st.lists(st.booleans(), min_size=1, max_size=20))
@settings(max_examples=50)
def test_nullable_bool_dtype(values):
    df = pd.DataFrame({'col': pd.array(values, dtype='boolean')})

    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result, df)

if __name__ == "__main__":
    test_nullable_bool_dtype()