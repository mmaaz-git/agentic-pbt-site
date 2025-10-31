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
print(f"hash(dtype): {hash(dtype)}")
print(f"hash(string_repr): {hash(string_repr)}")
print(f"hash(dtype) == hash(string_repr): {hash(dtype) == hash(string_repr)}")

# Dictionary lookup demonstration
d = {dtype: "value"}
print(f"\nd[dtype]: {d[dtype]}")
try:
    print(f"d[string_repr]: {d[string_repr]}")
except KeyError:
    print(f"d[string_repr]: KeyError - not found!")

# Show that despite being equal, they behave differently in sets
s = {dtype}
print(f"\nstring_repr in set containing dtype: {string_repr in s}")
print(f"But dtype == string_repr: {dtype == string_repr}")

# This violates Python's requirement that if a == b, then hash(a) == hash(b)
print(f"\nPython hash-equality contract violation:")
print(f"  dtype == string_repr: {dtype == string_repr}")
print(f"  hash(dtype) == hash(string_repr): {hash(dtype) == hash(string_repr)}")
print(f"  This violates the requirement that equal objects must have equal hashes!")