from hypothesis import given, settings, strategies as st
from numpy.polynomial import Polynomial
import numpy as np

polynomial_coefs = st.lists(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    min_size=1,
    max_size=6
)

@given(polynomial_coefs)
@settings(max_examples=5)  # Reduced for testing
def test_cast_to_same_type(coefs):
    p = Polynomial(coefs)
    try:
        p_cast = p.cast(Polynomial)
        assert np.allclose(p.coef, p_cast.coef, rtol=1e-10, atol=1e-10)
        print(f"Success with coefs: {coefs}")
        return True
    except AttributeError as e:
        print(f"Failed with coefs {coefs}: AttributeError: {e}")
        return False
    except Exception as e:
        print(f"Failed with coefs {coefs}: {type(e).__name__}: {e}")
        return False

# Run the test
print("Running hypothesis test:")
test_cast_to_same_type()

# Test the specific failing case mentioned
print("\nTesting specific case [0.0]:")
p = Polynomial([0.0])
try:
    p_cast = p.cast(Polynomial)
    print(f"Cast successful: {p_cast}")
except AttributeError as e:
    print(f"AttributeError: {e}")