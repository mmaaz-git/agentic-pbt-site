import warnings
import numpy.typing as npt

print("Accessing npt.NBitBase (should emit deprecation warning but doesn't):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Warnings raised: {len(w)}")

print("\nCalling __getattr__ directly (emits warning as intended):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__("NBitBase")
    print(f"Warnings raised: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")

print("\nChecking if NBitBase is in module __dict__:")
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")