"""
Simulate the fix to verify it would work.

This test demonstrates that if NBitBase were not directly imported,
the __getattr__ method would be called and emit the warning.
"""
import warnings

# Create a mock module to simulate numpy.typing
import types
import sys

# Create the mock module
mock_module = types.ModuleType('mock_numpy_typing')

# Import NBitBase but DON'T add it to the module namespace directly
from numpy._typing import NBitBase as _NBitBase

# Set up __all__ without NBitBase initially
mock_module.__all__ = ["ArrayLike", "DTypeLike", "NDArray"]

# Add the __getattr__ function that would handle NBitBase
def mock_getattr(name):
    if name == "NBitBase":
        import warnings
        warnings.warn(
            "`NBitBase` is deprecated and will be removed from numpy.typing in the "
            "future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper "
            "bound, instead. (deprecated in NumPy 2.3)",
            DeprecationWarning,
            stacklevel=2,
        )
        return _NBitBase
    raise AttributeError(f"module has no attribute {name!r}")

mock_module.__getattr__ = mock_getattr

# Test the mock module
print("Testing simulated fix...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Access NBitBase through the module
    result = mock_module.__getattr__("NBitBase")

    if len(w) == 1:
        print("SUCCESS: Deprecation warning was emitted!")
        print(f"  Warning: {w[0].category.__name__}: {w[0].message}")
    else:
        print(f"FAILED: {len(w)} warnings emitted (expected 1)")

print("\nVerifying that NBitBase was returned correctly...")
print(f"  Returned type: {result}")
print(f"  Is NBitBase: {result is _NBitBase}")