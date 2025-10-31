from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import tempfile
import os


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-1000, max_value=1000),
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=50)
        ),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=200)
def test_round_trip_write_read(data):
    df = pd.DataFrame(data, columns=['int_col', 'float_col', 'str_col'])

    # Filter out illegal characters that Excel doesn't support
    for idx, row in enumerate(data):
        text = row[2]
        # Skip data with illegal Excel characters (control characters)
        if any(ord(ch) < 32 and ch not in '\t\n\r' for ch in text):
            assume(False)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_excel(tmp_path, index=False, engine='openpyxl')
        df_read = pd.read_excel(tmp_path, engine='openpyxl')

        assert len(df_read) == len(df), f"Row count mismatch: expected {len(df)}, got {len(df_read)}"
        assert list(df_read.columns) == list(df.columns)

        pd.testing.assert_frame_equal(df, df_read, check_dtype=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    # Run the property-based test
    test_round_trip_write_read()