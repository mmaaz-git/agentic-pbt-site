import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat
from hypothesis import given, strategies as st, settings

@settings(max_examples=50)
@given(
    varname=st.text(
        alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')),
        min_size=1,
        max_size=10
    ),
    appendmat=st.booleans(),
)
def test_appendmat_parameter(varname, appendmat):
    with tempfile.TemporaryDirectory() as tmpdir:
        if appendmat:
            filename = os.path.join(tmpdir, 'test')
            expected_file = os.path.join(tmpdir, 'test.mat')
        else:
            filename = os.path.join(tmpdir, 'test.mat')
            expected_file = filename

        arr = np.array([1, 2, 3])
        data = {varname: arr}

        savemat(filename, data, appendmat=appendmat)

        assert os.path.exists(expected_file), f"Expected file {expected_file} does not exist (appendmat={appendmat}, files={os.listdir(tmpdir)})"

# Run the test
print("Running property-based test...")
try:
    test_appendmat_parameter()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")