import numpy.typing as npt

# Check if NBitBase is in the module's __dict__
print("NBitBase in npt.__dict__:", "NBitBase" in npt.__dict__)
print("NBitBase in dir(npt):", "NBitBase" in dir(npt))

# Check if we can access it directly
print("Can access NBitBase:", hasattr(npt, "NBitBase"))

# Test __getattr__ directly (this won't normally be called)
print("\nTrying to call __getattr__ directly:")
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = npt.__getattr__("NBitBase")
    if len(w) > 0:
        print(f"Direct __getattr__ call triggered warning: {w[0].message}")
    else:
        print("Direct __getattr__ call did not trigger warning")