import io
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(
    text_data=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=['Cs', 'Cc']), min_size=0, max_size=20),
        min_size=1,
        max_size=10
    ),
    num_cols=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=200)
def test_engine_equivalence_text(text_data, num_cols):
    columns = [f'col{i}' for i in range(num_cols)]
    data = {col: text_data for col in columns}
    df = pd.DataFrame(data)
    csv_str = df.to_csv(index=False)

    df_c = pd.read_csv(io.StringIO(csv_str), engine='c')
    df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

    pd.testing.assert_frame_equal(df_c, df_python, check_dtype=True)

if __name__ == "__main__":
    # Run the test
    test_engine_equivalence_text()