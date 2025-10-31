import pandas as pd
import io
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal


@given(st.data())
@settings(max_examples=200)
def test_table_orient_roundtrip(data):
    num_rows = data.draw(st.integers(min_value=1, max_value=5))
    num_cols = data.draw(st.integers(min_value=1, max_value=5))

    columns = [f'col_{i}' for i in range(num_cols)]

    df_data = {}
    for col in columns:
        df_data[col] = data.draw(
            st.lists(
                st.integers(),
                min_size=num_rows,
                max_size=num_rows
            )
        )

    df = pd.DataFrame(df_data)

    json_str = df.to_json(orient='table')
    df_roundtrip = pd.read_json(io.StringIO(json_str), orient='table')

    assert_frame_equal(df, df_roundtrip, check_dtype=False)


if __name__ == "__main__":
    test_table_orient_roundtrip()