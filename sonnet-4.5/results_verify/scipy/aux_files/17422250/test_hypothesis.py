from hypothesis import given, strategies as st, assume
import tempfile
import os
from scipy.io.matlab import savemat

@given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
       st.booleans())
def test_appendmat_behavior(base_name, appendmat):
    assume('.' not in base_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, base_name)
        data = {'x': 1.0}

        savemat(fname, data, appendmat=appendmat)

        if appendmat:
            expected_fname = fname + '.mat'
            assert os.path.exists(expected_fname), f"Expected file {expected_fname} not found"
        else:
            assert os.path.exists(fname), f"Expected file {fname} not found"

# Test with specific failing case
def test_specific_case():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test')
        data = {'x': 1.0}

        savemat(fname, data, appendmat=True)

        print(f"Files in directory: {os.listdir(tmpdir)}")
        print(f"Expected 'test.mat': {os.path.exists(fname + '.mat')}")
        print(f"Actual 'test' exists: {os.path.exists(fname)}")

        assert os.path.exists(fname + '.mat'), "Expected file 'test.mat' not found"

if __name__ == "__main__":
    # Run the specific test case
    try:
        test_specific_case()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run hypothesis test with explicit example
    try:
        test_appendmat_behavior('test', True)
        print("Hypothesis test passed!")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")