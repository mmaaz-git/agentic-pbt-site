from hypothesis import given, strategies as st
from scipy.io.arff._arffread import NominalAttribute
import numpy as np
import pytest

@given(name=st.text(min_size=1, max_size=20))
def test_nominal_str_should_handle_empty_values(name):
    """__str__ should handle empty values list gracefully"""
    attr = NominalAttribute.__new__(NominalAttribute)
    attr.name = name
    attr.values = ()
    attr.dtype = np.bytes_
    attr.range = ()
    attr.type_name = 'nominal'

    try:
        result = str(attr)
        assert isinstance(result, str)
    except IndexError:
        pytest.fail("__str__ should not raise IndexError on empty values")

if __name__ == "__main__":
    test_nominal_str_should_handle_empty_values()