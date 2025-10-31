import pandas as pd
from hypothesis import given, settings
from hypothesis.extra import pandas as pdst


@given(pdst.data_frames(columns=[
    pdst.column('A', dtype=int),
    pdst.column('B', dtype=int)
]))
@settings(max_examples=500)
def test_transpose_involution_preserves_dtype(df):
    result = df.T.T
    for col in df.columns:
        assert df[col].dtype == result[col].dtype, f"Column {col} dtype changed from {df[col].dtype} to {result[col].dtype}"

if __name__ == "__main__":
    test_transpose_involution_preserves_dtype()
    print("Test completed without errors")