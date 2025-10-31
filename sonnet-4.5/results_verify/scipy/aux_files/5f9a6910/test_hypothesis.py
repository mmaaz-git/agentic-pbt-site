from hypothesis import given, strategies as st, settings
from scipy.io.matlab import savemat, loadmat
import numpy as np
import tempfile
import os

@st.composite
def valid_varnames(draw):
    first_char = draw(st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    rest = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_'),
        min_size=0, max_size=10
    ))
    return first_char + rest

@given(
    varname=valid_varnames(),
    value=st.floats(min_value=-1e10, max_value=1e10,
                   allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_valid_varnames_roundtrip(varname, value):
    arr = np.array([[value]])
    mdict = {varname: arr}

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        savemat(fname, mdict, format='5')
        loaded = loadmat(fname)
        assert varname in loaded
    finally:
        if os.path.exists(fname):
            os.unlink(fname)

if __name__ == "__main__":
    # Test the specific failing case mentioned
    print("Testing specific failing case: varname='aĀ'")

    # Direct test without hypothesis decorator
    arr = np.array([[1.0]])
    mdict = {'aĀ': arr}

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        fname = f.name

    try:
        savemat(fname, mdict, format='5')
        loaded = loadmat(fname)
        print(f"Success! Loaded: {loaded}")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(fname):
            os.unlink(fname)