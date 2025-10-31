from hypothesis import given, strategies as st
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
def test_extensiondtype_string_equality_implies_hash_equality(param):
    dtype = ParametrizedDtype(param)
    string_repr = dtype.name

    assert dtype == string_repr
    assert hash(dtype) == hash(string_repr)


if __name__ == "__main__":
    test_extensiondtype_string_equality_implies_hash_equality()