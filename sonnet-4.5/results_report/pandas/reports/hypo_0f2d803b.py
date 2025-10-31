import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.interchange import from_dataframe

@given(
    data=st.lists(st.integers(), min_size=0, max_size=100),
    col_name=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
)
@settings(max_examples=1000)
def test_round_trip_integer_column(data, col_name):
    df = pd.DataFrame({col_name: data})

    interchange_df = df.__dataframe__()
    df_roundtrip = from_dataframe(interchange_df)

    assert df.equals(df_roundtrip), f"Round-trip failed: {df} != {df_roundtrip}"

# Run the test
test_round_trip_integer_column()