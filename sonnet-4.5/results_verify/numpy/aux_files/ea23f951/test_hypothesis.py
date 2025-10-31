import tempfile
import os
import numpy as np
from scipy import io
from hypothesis import given, strategies as st, settings


def make_matlab_compatible_dict():
    valid_name = st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=10
    ).filter(lambda x: x[0].isalpha() and x.isidentifier())

    value_strategy = st.one_of(
        st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
        st.integers(min_value=-1000000, max_value=1000000),
        st.lists(st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False),
                 min_size=0, max_size=10),
    )

    return st.dictionaries(valid_name, value_strategy, min_size=1, max_size=5)


@given(make_matlab_compatible_dict())
@settings(max_examples=50)
def test_savemat_loadmat_roundtrip(data_dict):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        io.savemat(fname, data_dict)
        loaded = io.loadmat(fname)

        for key in data_dict:
            assert key in loaded
            original_val = np.array(data_dict[key])
            loaded_val = loaded[key]

            if original_val.ndim == 0:
                original_val = np.atleast_2d(original_val)

            np.testing.assert_array_almost_equal(original_val, loaded_val)
    finally:
        if os.path.exists(fname):
            os.unlink(fname)

if __name__ == "__main__":
    test_savemat_loadmat_roundtrip()