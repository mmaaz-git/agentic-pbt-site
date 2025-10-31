from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.api.extensions import ExtensionDtype


class ParametrizedDtype(ExtensionDtype):
    _metadata = ('param',)

    def __init__(self, param=0):
        self.param = param

    @property
    def name(self):
        return f"param[{self.param}]"

    @classmethod
    def construct_from_string(cls, string):
        if string.startswith("param[") and string.endswith("]"):
            param_str = string[6:-1]
            try:
                param = int(param_str)
                return cls(param)
            except ValueError:
                pass
        raise TypeError(f"Cannot construct from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return np.ndarray

    @property
    def type(self):
        return object


@given(st.integers(-1000, 1000))
@settings(max_examples=10)  # Limiting to 10 examples for quick testing
def test_extensiondtype_string_equality_implies_hash_equality(param):
    dtype = ParametrizedDtype(param)
    string_repr = dtype.name

    print(f"Testing param={param}")
    print(f"  dtype == string_repr: {dtype == string_repr}")
    print(f"  hash(dtype) == hash(string_repr): {hash(dtype) == hash(string_repr)}")

    assert dtype == string_repr, f"dtype should equal its string representation for param={param}"
    assert hash(dtype) == hash(string_repr), f"hash(dtype) should equal hash(string_repr) when they are equal for param={param}"


# Run the test
try:
    test_extensiondtype_string_equality_implies_hash_equality()
    print("\nAll tests passed!")
except AssertionError as e:
    print(f"\nTest failed: {e}")