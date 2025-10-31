from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile

def make_odr():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    data = Data(x, y)
    return ODR(data, unilinear, beta0=[1.0, 0.0])

@given(
    init=st.integers(min_value=-5, max_value=10),
    iter_param=st.integers(min_value=-5, max_value=10),
    final=st.integers(min_value=-5, max_value=10)
)
@settings(max_examples=200)
def test_set_iprint_validates_inputs(init, iter_param, final):
    odr_obj = make_odr()
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        odr_obj.rptfile = f.name

    try:
        odr_obj.set_iprint(init=init, iter=iter_param, final=final)
    except ValueError as e:
        if "is not in list" in str(e):
            raise AssertionError(f"Missing input validation: {e}")

# Run the test
print("Running property-based test...")
try:
    test_set_iprint_validates_inputs()
    print("Test passed (no issues found)")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")