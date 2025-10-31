import warnings
import numpy.typing as npt

print("Test 1: Accessing NBitBase through npt.NBitBase")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"  Number of warnings: {len(w)}")
    if len(w) > 0:
        print(f"  Warning category: {w[0].category}")
        print(f"  Warning message: {w[0].message}")
    else:
        print("  No warnings captured")

print("\nTest 2: Accessing NBitBase through npt.__getattr__")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__('NBitBase')
    print(f"  Number of warnings: {len(w)}")
    if len(w) > 0:
        print(f"  Warning category: {w[0].category}")
        print(f"  Warning message: {w[0].message}")

print("\nTest 3: Check if NBitBase is in __all__")
print(f"  NBitBase in __all__: {'NBitBase' in npt.__all__}")

print("\nTest 4: Check if NBitBase is in globals")
import numpy.typing
print(f"  NBitBase in module globals: {'NBitBase' in dir(numpy.typing)}")

print("\nTest 5: Check if NBitBase is directly imported")
print(f"  NBitBase defined in module: {hasattr(numpy.typing, 'NBitBase')}")