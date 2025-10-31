import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.io import parsers
from io import StringIO
from hypothesis import given, strategies as st, settings


@st.composite
def csv_dataframes(draw):
    num_rows = draw(st.integers(min_value=0, max_value=20))
    num_cols = draw(st.integers(min_value=1, max_value=10))
    col_names = [f"col{i}" for i in range(num_cols)]

    data = {}
    for col in col_names:
        col_type = draw(st.sampled_from(['int', 'float', 'str', 'bool']))
        if col_type == 'str':
            data[col] = draw(st.lists(st.text(min_size=0, max_size=20), min_size=num_rows, max_size=num_rows))
        else:
            data[col] = draw(st.lists(st.integers(), min_size=num_rows, max_size=num_rows))

    return pd.DataFrame(data)


@given(csv_dataframes())
@settings(max_examples=50)
def test_engine_consistency(df):
    csv_string = df.to_csv(index=False)

    df_c = parsers.read_csv(StringIO(csv_string), engine='c')
    df_python = parsers.read_csv(StringIO(csv_string), engine='python')

    assert df_c.shape == df_python.shape
    pd.testing.assert_frame_equal(df_c, df_python)

if __name__ == "__main__":
    # Run the test
    test_engine_consistency()