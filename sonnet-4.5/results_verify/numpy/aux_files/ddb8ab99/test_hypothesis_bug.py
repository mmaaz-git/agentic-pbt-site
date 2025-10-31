import pandas as pd
import io
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal


@given(
    st.data(),
    st.sampled_from(['index', 'columns'])
)
@settings(max_examples=200)
def test_json_roundtrip_preserves_string_axes(data, orient):
    num_rows = data.draw(st.integers(min_value=1, max_value=5))
    num_cols = data.draw(st.integers(min_value=1, max_value=5))

    df = pd.DataFrame({
        f'col_{i}': [j for j in range(num_rows)]
        for i in range(num_cols)
    })

    if orient == 'index':
        df.index = df.index.astype(str)
    else:
        df.columns = df.columns.astype(str)

    json_str = df.to_json(orient=orient)
    df_roundtrip = pd.read_json(io.StringIO(json_str), orient=orient)

    assert_frame_equal(df, df_roundtrip, check_dtype=False)

if __name__ == "__main__":
    test_json_roundtrip_preserves_string_axes()
    print("Test passed!")