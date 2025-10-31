import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.interchange.from_dataframe import from_dataframe

@given(st.data())
@settings(max_examples=200)
def test_from_dataframe_with_unicode_strings(data):
    n_rows = data.draw(st.integers(min_value=1, max_value=20))

    values = []
    for _ in range(n_rows):
        val = data.draw(st.text(
            alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0x1FFFF),
            min_size=0,
            max_size=20
        ))
        values.append(val)

    df_original = pd.DataFrame({'col': values})
    interchange_obj = df_original.__dataframe__()
    df_roundtrip = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(df_original, df_roundtrip)

if __name__ == "__main__":
    test_from_dataframe_with_unicode_strings()