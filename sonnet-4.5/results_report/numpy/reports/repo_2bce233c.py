import warnings
import numpy.typing as npt

# Test 1: Normal attribute access (expected to trigger warning but doesn't)
print("Test 1: Normal attribute access")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = npt.NBitBase
    print(f"Warnings captured: {len(w)}")
    print(f"Expected: 1, Actual: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")
    else:
        print("No warning triggered!")

print("\n" + "="*50 + "\n")

# Test 2: Verify NBitBase exists and is accessible
print("Test 2: Verify NBitBase exists")
print(f"NBitBase type: {type(npt.NBitBase)}")
print(f"NBitBase is in module __dict__: {'NBitBase' in npt.__dict__}")
print(f"NBitBase is in __all__: {'NBitBase' in npt.__all__}")

print("\n" + "="*50 + "\n")

# Test 3: Direct call to __getattr__ (this SHOULD work)
print("Test 3: Direct __getattr__ call")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = npt.__getattr__("NBitBase")
    print(f"Warnings captured: {len(w)}")
    print(f"Expected: 1, Actual: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")
        print(f"Warning category: {w[0].category}")