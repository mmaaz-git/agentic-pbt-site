from hypothesis import given, strategies as st, settings
import pandas as pd
from io import StringIO


@given(
    columns=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5, unique=True),
    num_rows=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=200)
def test_json_roundtrip_column_names(columns, num_rows):
    data = {col: list(range(num_rows)) for col in columns}
    df = pd.DataFrame(data)

    json_str = df.to_json(orient='split')
    result = pd.read_json(StringIO(json_str), orient='split')

    assert list(result.columns) == list(df.columns)

if __name__ == "__main__":
    test_json_roundtrip_column_names()