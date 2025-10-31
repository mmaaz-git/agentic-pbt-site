import os
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning
import numpy as np
import tempfile
import warnings

@given(
    value=st.floats(min_value=-1e10, max_value=1e10,
                   allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_digit_start_vars_not_saved(value):
    arr = np.array([[value]])
    mdict = {'1invalid': arr, 'valid': arr}

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(fname, mdict, format='5')
            warning_issued = any(issubclass(warn.category, MatWriteWarning) for warn in w)

        loaded = loadmat(fname)

        assert 'valid' in loaded
        assert '1invalid' not in loaded, "Variables starting with digit should not be saved"
        assert warning_issued, "MatWriteWarning should be issued for digit-starting variables"
    finally:
        if os.path.exists(fname):
            os.unlink(fname)

if __name__ == "__main__":
    # Run the test
    test_digit_start_vars_not_saved()
    print("All tests passed!")