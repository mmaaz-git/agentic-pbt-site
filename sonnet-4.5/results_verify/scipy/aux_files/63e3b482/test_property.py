from hypothesis import given, strategies as st, settings, example
import numpy as np
from io import BytesIO
import scipy.io.matlab as sio

@given(st.text(min_size=1, max_size=20))
@settings(max_examples=1000)
@example('Ā')  # The specific failing case mentioned
def test_savemat_variable_name_encoding(varname):
    data = {varname: np.array([[1.0]])}
    f = BytesIO()
    try:
        sio.savemat(f, data)
        f.seek(0)
        result = sio.loadmat(f)
        assert varname in result or varname.startswith('_')
        print(f"✓ Variable name '{varname}' worked")
    except UnicodeEncodeError as e:
        print(f"✗ UnicodeEncodeError for variable name '{varname}': {e}")

if __name__ == "__main__":
    test_savemat_variable_name_encoding()