import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float)
]))
@settings(max_examples=200)
def test_to_dict_tight_from_dict_tight_roundtrip(df):
    dict_repr = df.to_dict(orient='tight')
    result = pd.DataFrame.from_dict(dict_repr, orient='tight')
    assert result.equals(df), f"Round-trip with orient='tight' failed. Original dtypes: {df.dtypes.to_dict()}, Result dtypes: {result.dtypes.to_dict()}"

if __name__ == "__main__":
    test_to_dict_tight_from_dict_tight_roundtrip()