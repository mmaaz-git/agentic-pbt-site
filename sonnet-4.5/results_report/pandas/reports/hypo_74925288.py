from hypothesis import given, settings, strategies as st
import pandas as pd
from io import StringIO

@given(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e10,
        max_value=1e10,
        width=64
    )
)
@settings(max_examples=500)
def test_dataframe_json_roundtrip(value):
    df = pd.DataFrame({'col': [value]})
    json_str = df.to_json(orient='records')
    df_restored = pd.read_json(StringIO(json_str), orient='records')

    orig = df['col'].iloc[0]
    restored = df_restored['col'].iloc[0]

    assert orig == restored, f"Round-trip failed: {orig} != {restored}"

if __name__ == "__main__":
    test_dataframe_json_roundtrip()