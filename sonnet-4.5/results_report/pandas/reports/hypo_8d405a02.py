import pandas as pd
import io
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=10))
@settings(max_examples=200)
def test_csv_handles_special_chars_in_column_names(name):
    df = pd.DataFrame([[1]], columns=[name])
    csv_str = df.to_csv(index=False)
    result = pd.read_csv(io.StringIO(csv_str))

    assert len(result.columns) == 1
    assert result.columns[0] == name

if __name__ == "__main__":
    test_csv_handles_special_chars_in_column_names()