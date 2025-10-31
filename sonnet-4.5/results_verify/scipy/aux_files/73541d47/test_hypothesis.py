import tempfile
import numpy as np
from scipy import io as scipy_io
from hypothesis import given, strategies as st, settings


@st.composite
def dense_real_arrays(draw):
    rows = draw(st.integers(min_value=1, max_value=20))
    cols = draw(st.integers(min_value=1, max_value=20))
    data = draw(
        st.lists(
            st.floats(
                min_value=-1e10,
                max_value=1e10,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=rows * cols,
            max_size=rows * cols,
        )
    )
    return np.array(data).reshape(rows, cols)


@given(dense_real_arrays())
@settings(max_examples=10, deadline=None)
def test_mminfo_consistency_with_mmread_dense(original):
    print(f"Testing array of shape {original.shape}")
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.mtx', delete=False) as f:
        scipy_io.mmwrite(f, original)
        f.flush()
        f.seek(0)

        info_result = scipy_io.mminfo(f)
        rows, cols, entries, format_type, field = info_result[:5]

        f.seek(0)
        result = scipy_io.mmread(f)

    assert rows == original.shape[0]
    assert cols == original.shape[1]
    print(f"  - Test passed for shape {original.shape}")


if __name__ == "__main__":
    print("Running hypothesis test...")
    test_mminfo_consistency_with_mmread_dense()