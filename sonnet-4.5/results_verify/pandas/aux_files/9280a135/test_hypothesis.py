import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, column


@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float),
    column('c', dtype=str),
]))
@settings(max_examples=100)
def test_column_names_preserved(df):
    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    assert list(interchange_obj.column_names()) == list(result.columns)

if __name__ == "__main__":
    test_column_names_preserved()
    print("Property-based test completed")