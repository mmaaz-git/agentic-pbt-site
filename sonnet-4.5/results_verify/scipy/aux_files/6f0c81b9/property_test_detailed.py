from hypothesis import given, strategies as st, settings, example
import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile
import traceback

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
@settings(max_examples=50)  # Reduced for faster debugging
@example(init=3, iter_param=0, final=0)  # Force one failing case
def test_set_iprint_validates_inputs(init, iter_param, final):
    odr_obj = make_odr()
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        odr_obj.rptfile = f.name

    try:
        odr_obj.set_iprint(init=init, iter=iter_param, final=final)
    except ValueError as e:
        error_msg = str(e)
        if "is not in list" in error_msg:
            print(f"Found invalid error for init={init}, iter={iter_param}, final={final}: {error_msg}")
            raise AssertionError(f"Missing input validation - got internal error: {e}")
        else:
            # This could be a proper validation error
            pass

# Run the test
print("Running property-based test with details...")
try:
    test_set_iprint_validates_inputs()
    print("Test passed - no issues found")
except Exception as e:
    print(f"Test failed with: {e}")
    traceback.print_exc()