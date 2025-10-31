import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings

@given(
    n_rows=st.integers(min_value=1, max_value=20),
    n_cols=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=200)
def test_transpose_preserves_dtypes(n_rows, n_cols):
    dtypes_dict = {f'col_{i}': np.float64 if i % 2 == 0 else np.int64 for i in range(n_cols)}

    data = {}
    for col, dtype in dtypes_dict.items():
        if dtype == np.float64:
            data[col] = np.random.randn(n_rows)
        else:
            data[col] = np.random.randint(0, 100, n_rows)

    df = pd.DataFrame(data)
    df_t = df.T
    df_tt = df_t.T

    assert df.shape == df_tt.shape

    for col in df.columns:
        assert df[col].dtype == df_tt[col].dtype  # FAILS: int64 becomes float64

# Run the test
if __name__ == "__main__":
    test_transpose_preserves_dtypes()