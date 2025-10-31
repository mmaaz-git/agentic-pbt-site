import pandas as pd
from io import StringIO
from hypothesis import given, settings, strategies as st
from pandas.testing import assert_frame_equal

@st.composite
def dataframes(draw):
    num_rows = draw(st.integers(min_value=0, max_value=20))
    num_cols = draw(st.integers(min_value=1, max_value=10))
    columns = [f"col_{i}" for i in range(num_cols)]
    data = {col: draw(st.lists(st.integers(), min_size=num_rows, max_size=num_rows))
            for col in columns}
    return pd.DataFrame(data)

@given(dataframes())
@settings(max_examples=200)
def test_dataframe_roundtrip_split(df):
    json_str = df.to_json(orient='split')
    df_roundtrip = pd.read_json(StringIO(json_str), orient='split')
    assert_frame_equal(df, df_roundtrip, check_dtype=False)

if __name__ == "__main__":
    # Test the specific failing case
    try:
        df = pd.DataFrame({"col_0": []})
        json_str = df.to_json(orient='split')
        df_roundtrip = pd.read_json(StringIO(json_str), orient='split')
        assert_frame_equal(df, df_roundtrip, check_dtype=False)
        print("Specific test case passed (with check_dtype=False)")
    except Exception as e:
        print(f"Specific test case failed: {e}")

    # Run hypothesis tests
    print("\nRunning hypothesis tests...")
    test_dataframe_roundtrip_split()