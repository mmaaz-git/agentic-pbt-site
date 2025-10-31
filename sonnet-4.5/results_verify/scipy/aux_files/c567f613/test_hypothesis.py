from hypothesis import given, settings, strategies as st
import numpy as np
import tempfile
import os
import warnings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

@given(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122)))
@settings(max_examples=10)  # Reduced for testing
def test_variable_names_starting_with_digit(name):
    varname = '0' + name
    arr = np.array([1, 2, 3])

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        mdict = {varname: arr}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(fname, mdict)
            if len(w) > 0:
                assert issubclass(w[0].category, MatWriteWarning), f"Expected MatWriteWarning, got {w[0].category}"

        result = loadmat(fname)
        assert varname not in result or len(result[varname]) == 0, f"Variable {varname} should not be saved but it was!"
    finally:
        if os.path.exists(fname):
            os.unlink(fname)

# Run the test
if __name__ == "__main__":
    test_variable_names_starting_with_digit()
    print("Test completed")