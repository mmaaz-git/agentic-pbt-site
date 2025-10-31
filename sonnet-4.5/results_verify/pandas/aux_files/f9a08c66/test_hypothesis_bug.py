from hypothesis import given, strategies as st
import numpy as np
from pandas.core.dtypes.common import ensure_python_int

@given(st.floats(allow_nan=False, allow_infinity=False).filter(lambda x: x == int(x)))
def test_ensure_python_int_type_signature_contract(value):
    """
    The type signature says: value: int | np.integer
    But the function accepts floats that equal their integer conversion.
    This violates the type contract.
    """
    try:
        result = ensure_python_int(value)
        print(f"Function accepted float {value} and returned {result}")
        assert False, f"Function with signature 'int | np.integer' accepted float {value}"
    except TypeError:
        pass

if __name__ == "__main__":
    test_ensure_python_int_type_signature_contract()