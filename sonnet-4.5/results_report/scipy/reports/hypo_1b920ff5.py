from hypothesis import given, strategies as st, assume, example
import tempfile
import os
from scipy.io.matlab import savemat

@given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
       st.booleans())
@example('test', True)  # Known failing example
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

# Run the test
if __name__ == "__main__":
    # Run hypothesis test
    print("Running Hypothesis test with known failing input base_name='test', appendmat=True:")
    try:
        test_appendmat_behavior()
    except AssertionError as e:
        print(f"Falsifying example: base_name='test', appendmat=True")
        print(f"AssertionError: {e}")