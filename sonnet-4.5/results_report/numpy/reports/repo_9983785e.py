import warnings
import numpy.typing as npt

# Test if NBitBase deprecation warning is triggered
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    # Try to access NBitBase
    result = npt.NBitBase

    # Check if any warnings were triggered
    if len(w) == 0:
        print("BUG: No deprecation warning triggered when accessing numpy.typing.NBitBase")
        print(f"Successfully accessed NBitBase: {result}")
    else:
        print(f"OK: Warning triggered")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")

# Additional verification - show that NBitBase is directly in module namespace
print(f"\n'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"'NBitBase' in npt.__all__: {'NBitBase' in npt.__all__}")

# Show that __getattr__ is never called for NBitBase
print("\nDirect __getattr__ call (should trigger warning):")
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    try:
        result2 = npt.__getattr__("NBitBase")
        if len(w2) > 0:
            print(f"  Warning triggered via __getattr__: {w2[0].message}")
    except AttributeError as e:
        print(f"  AttributeError: {e}")