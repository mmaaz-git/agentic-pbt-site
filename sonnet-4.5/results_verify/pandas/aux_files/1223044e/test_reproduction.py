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


dtype = ParametrizedDtype(0)
string_repr = dtype.name

print(f"dtype == string_repr: {dtype == string_repr}")
print(f"hash(dtype) == hash(string_repr): {hash(dtype) == hash(string_repr)}")

d = {dtype: "value"}
print(f"d[dtype]: {d[dtype]}")
print(f"d[string_repr]: {d.get(string_repr, 'KeyError - not found!')}")

# Additional tests to verify the behavior
print("\nAdditional tests:")
print(f"dtype name: {dtype.name}")
print(f"string_repr: {string_repr}")
print(f"hash(dtype): {hash(dtype)}")
print(f"hash(string_repr): {hash(string_repr)}")

# Test if the reconstructed dtype from string equals the original
reconstructed = ParametrizedDtype.construct_from_string(string_repr)
print(f"\nreconstructed == dtype: {reconstructed == dtype}")
print(f"hash(reconstructed): {hash(reconstructed)}")
print(f"hash(reconstructed) == hash(dtype): {hash(reconstructed) == hash(dtype)}")