import io
import pandas as pd
from hypothesis import given, strategies as st, settings


@given(
    data=st.lists(
        st.lists(
            st.text(alphabet=' ', min_size=1, max_size=5),
            min_size=1,
            max_size=3
        ),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=50)
def test_roundtrip_whitespace_dataframe(data):
    num_cols = len(data[0])
    if not all(len(row) == num_cols for row in data):
        return

    col_names = [f"col{i}" for i in range(num_cols)]
    df = pd.DataFrame(data, columns=col_names)

    csv_str = df.to_csv(index=False)
    df_roundtrip = pd.read_csv(io.StringIO(csv_str))

    assert len(df) == len(df_roundtrip), \
        f"Row count changed: {len(df)} -> {len(df_roundtrip)}"

if __name__ == "__main__":
    # Run the test to see if it fails
    try:
        test_roundtrip_whitespace_dataframe()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")