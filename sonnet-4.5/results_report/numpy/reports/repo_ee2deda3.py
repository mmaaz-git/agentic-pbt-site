import warnings
import numpy.typing as npt

# Attempting to catch the deprecation warning that should be emitted
# when accessing NBitBase (deprecated in NumPy 2.3)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Access NBitBase - this should trigger a deprecation warning
    obj = npt.NBitBase

    # Check if any warnings were caught
    print(f"Warnings caught: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")

    if len(w) > 0:
        for warning in w:
            print(f"Warning category: {warning.category}")
            print(f"Warning message: {warning.message}")
    else:
        print("No warnings were emitted (BUG: deprecation warning not triggered)")

    # Verify that NBitBase is accessible
    print(f"\nNBitBase object type: {type(obj)}")
    print(f"NBitBase object: {obj}")