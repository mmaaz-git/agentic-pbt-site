import math
from hypothesis import given, settings, strategies as st
import scipy.constants as sc
import pytest

all_keys = list(sc.physical_constants.keys())

@given(st.sampled_from(all_keys))
@settings(max_examples=500)
def test_precision_calculation(key):
    result = sc.precision(key)
    value_const, unit_const, abs_precision = sc.physical_constants[key]

    if value_const == 0:
        pytest.skip("Cannot compute relative precision for zero value")

    expected = abs(abs_precision / value_const)
    assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-15)

if __name__ == "__main__":
    # Run the test
    test_precision_calculation()