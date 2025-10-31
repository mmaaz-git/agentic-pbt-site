import sys
import warnings

# Create a mock module to test __getattr__ behavior
class TestModule:
    def __init__(self):
        self.__all__ = ["NBitBase"]
        self.__DIR = self.__all__
        self.__DIR_SET = frozenset(self.__DIR)

    def __dir__(self):
        return self.__DIR

    def __getattr__(self, name):
        if name == "NBitBase":
            import warnings
            warnings.warn(
                "`NBitBase` is deprecated and will be removed from numpy.typing in the "
                "future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper "
                "bound, instead. (deprecated in NumPy 2.3)",
                DeprecationWarning,
                stacklevel=2,
            )
            return "NBitBase_value"
        raise AttributeError(f"module has no attribute {name!r}")

# Test without direct attribute
test_mod = TestModule()
print("Testing __getattr__ mechanism:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    value = test_mod.NBitBase
    print(f"  Value: {value}")
    print(f"  Warnings emitted: {len(w)}")
    if w:
        print(f"  Warning message: {w[0].message}")

# Now add NBitBase directly to bypass __getattr__
print("\nTesting with direct attribute (simulating the bug):")
test_mod.NBitBase = "NBitBase_direct"
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    value = test_mod.NBitBase
    print(f"  Value: {value}")
    print(f"  Warnings emitted: {len(w)}")
    if w:
        print(f"  Warning message: {w[0].message}")