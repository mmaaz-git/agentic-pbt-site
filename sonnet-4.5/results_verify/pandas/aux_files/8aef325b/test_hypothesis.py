from hypothesis import given, strategies as st, settings, reproduce_failure
import pandas as pd
import tempfile
import os

@given(st.data())
@settings(max_examples=30)
def test_string_roundtrip(data):
    """String DataFrames should round-trip through Excel"""
    n_rows = data.draw(st.integers(min_value=1, max_value=50))
    n_cols = data.draw(st.integers(min_value=1, max_value=5))

    df_data = {}
    for i in range(n_cols):
        col_name = f'col_{i}'
        df_data[col_name] = data.draw(st.lists(
            st.text(min_size=0, max_size=100),
            min_size=n_rows,
            max_size=n_rows
        ))

    df_original = pd.DataFrame(df_data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_original.to_excel(tmp_path, index=False)
        df_read = pd.read_excel(tmp_path)

        assert df_read.shape == df_original.shape, f"Shape mismatch: original {df_original.shape}, read back {df_read.shape}"
        pd.testing.assert_frame_equal(df_read, df_original, check_dtype=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    test_string_roundtrip()