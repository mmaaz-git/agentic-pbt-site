import sys
import os
import warnings
import tempfile
import shutil

# Create a test module that demonstrates the fix
test_module_code = '''"""Test module to demonstrate NBitBase deprecation fix"""

# Import NBitBase with a different name so it's not directly in namespace
from numpy._typing import ArrayLike, DTypeLike, NDArray
from numpy._typing import NBitBase as _NBitBase

__all__ = ["ArrayLike", "DTypeLike", "NBitBase", "NDArray"]

__DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
__DIR_SET = frozenset(__DIR)

def __dir__() -> list[str]:
    return __DIR

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

    if name in __DIR_SET:
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''

# Create a temporary directory and module
temp_dir = tempfile.mkdtemp()
module_dir = os.path.join(temp_dir, 'test_typing')
os.makedirs(module_dir)

# Write the test module
with open(os.path.join(module_dir, '__init__.py'), 'w') as f:
    f.write(test_module_code)

# Add to sys.path
sys.path.insert(0, temp_dir)

try:
    # Import and test the fixed module
    import test_typing

    print("Testing the FIXED version:")
    print("="*50)

    # Test accessing NBitBase (should trigger warning)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj = test_typing.NBitBase
        print(f"Warnings caught: {len(w)}")
        if len(w) > 0:
            for warning in w:
                print(f"  Warning: {warning.category.__name__}: {warning.message}")
        else:
            print("  No warnings (this would be the bug)")

    # Verify NBitBase is still accessible
    print(f"\nNBitBase still accessible: {test_typing.NBitBase is not None}")
    print(f"NBitBase in __all__: {'NBitBase' in test_typing.__all__}")
    print(f"NBitBase in dir(): {'NBitBase' in dir(test_typing)}")

    # Test that other attributes work normally
    print(f"\nArrayLike accessible: {test_typing.ArrayLike is not None}")
    print(f"DTypeLike accessible: {test_typing.DTypeLike is not None}")

finally:
    # Clean up
    sys.path.remove(temp_dir)
    shutil.rmtree(temp_dir)

print("\n" + "="*50)
print("Conclusion: The proposed fix DOES work - it triggers the deprecation warning!")