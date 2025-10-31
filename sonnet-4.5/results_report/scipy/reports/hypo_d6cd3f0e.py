import scipy.io.matlab as mat
import numpy as np
import tempfile
import os
from hypothesis import given, settings, strategies as st


@settings(max_examples=100)
@given(st.booleans())
def test_chars_as_strings_roundtrip(chars_as_strings):
    test_dict = {'text': 'hello world'}

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        temp_file1 = f.name
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        temp_file2 = f.name

    try:
        mat.savemat(temp_file1, test_dict)
        loaded1 = mat.loadmat(temp_file1, chars_as_strings=chars_as_strings)

        user_keys1 = {k: v for k, v in loaded1.items() if not k.startswith('__')}
        mat.savemat(temp_file2, user_keys1)
        loaded2 = mat.loadmat(temp_file2, chars_as_strings=chars_as_strings)

        user_keys2 = {k: v for k, v in loaded2.items() if not k.startswith('__')}

        for key in user_keys1:
            assert key in user_keys2
            assert np.array_equal(user_keys1[key], user_keys2[key]), \
                f"Arrays not equal for key '{key}': shape {user_keys1[key].shape} vs {user_keys2[key].shape}"

    finally:
        if os.path.exists(temp_file1):
            os.remove(temp_file1)
        if os.path.exists(temp_file2):
            os.remove(temp_file2)


if __name__ == "__main__":
    test_chars_as_strings_roundtrip()