"""Test if the proposed fix would work"""
import warnings

# Simulate the fix by creating a minimal module with __getattr__
import types
import sys

# Create a fake module to simulate the fix
fake_npt = types.ModuleType('fake_npt')

# Import NBitBase as _NBitBase (like the fix suggests)
from numpy._typing import NBitBase as _NBitBase

# Don't add NBitBase to __dict__ directly

# Add the __getattr__ function
def __getattr__(name: str):
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
    raise AttributeError(f"module fake_npt has no attribute {name!r}")

fake_npt.__getattr__ = __getattr__

# Test the fixed behavior
print("Testing the proposed fix:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = fake_npt.NBitBase

    print(f"Warnings caught: {len(w)}")
    if len(w) > 0:
        print(f"SUCCESS: Deprecation warning was emitted!")
        print(f"  Warning: {w[0].category.__name__}: {w[0].message}")
    else:
        print("FAIL: No warning emitted")

print(f"\n'NBitBase' in fake_npt.__dict__: {'NBitBase' in fake_npt.__dict__}")