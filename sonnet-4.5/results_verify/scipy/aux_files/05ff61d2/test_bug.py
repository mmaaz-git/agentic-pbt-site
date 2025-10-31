from io import BytesIO
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat
import warnings


@settings(max_examples=50)
@given(st.from_regex(r'^[0-9][a-zA-Z0-9_]*$', fullmatch=True))
def test_digit_key_not_saved(key):
    bio = BytesIO()
    data = {key: np.array([1, 2, 3])}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(bio, data)

        if len(w) > 0:
            assert any("MatWriteWarning" in str(warn.category) for warn in w)

    bio.seek(0)
    loaded = loadmat(bio)

    assert key not in loaded

if __name__ == "__main__":
    # Run the hypothesis test
    print("Running Hypothesis test...")
    try:
        test_digit_key_not_saved()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Error during test: {e}")