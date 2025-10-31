from hypothesis import given, strategies as st, settings
import pandas as pd
from io import StringIO

@given(
    data=st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.integers(),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=20,
    ),
    orient=st.sampled_from(['records', 'index', 'columns', 'values', 'split']),
)
@settings(max_examples=100)
def test_read_json_to_json_roundtrip_dataframe(data, orient):
    df = pd.DataFrame(data)
    json_str = df.to_json(orient=orient)
    df_back = pd.read_json(StringIO(json_str), orient=orient)

# Run the test
if __name__ == "__main__":
    test_read_json_to_json_roundtrip_dataframe()
    print("Test passed")