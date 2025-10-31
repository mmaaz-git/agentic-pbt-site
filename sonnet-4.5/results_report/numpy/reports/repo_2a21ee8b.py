import warnings
import numpy.typing as npt

# Test if accessing NBitBase emits a deprecation warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Access NBitBase
    _ = npt.NBitBase

    # Check for warnings
    print(f"Number of warnings emitted: {len(w)}")

    if len(w) == 0:
        print("ERROR: No deprecation warning was emitted when accessing NBitBase!")
        print("       NBitBase is documented as deprecated but no warning appears.")
    else:
        for warning in w:
            print(f"Warning category: {warning.category.__name__}")
            print(f"Warning message: {warning.message}")

    # Verify the attribute exists and is accessible
    print(f"\nNBitBase type: {type(npt.NBitBase)}")
    print(f"NBitBase module: {npt.NBitBase.__module__ if hasattr(npt.NBitBase, '__module__') else 'N/A'}")