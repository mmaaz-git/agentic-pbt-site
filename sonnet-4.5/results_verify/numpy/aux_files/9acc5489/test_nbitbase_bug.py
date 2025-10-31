import warnings
import numpy.typing as npt

print("Test 1: Direct access to NBitBase")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = npt.NBitBase

    print(f"Warnings caught: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")
    print(f"Actual: {len(w)} warnings")

    if len(w) == 0:
        print("BUG: No deprecation warning emitted!")
    else:
        for warning in w:
            print(f"  Warning: {warning.category.__name__}: {warning.message}")

print("\nTest 2: Using getattr")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = getattr(npt, "NBitBase")

    print(f"Warnings caught: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")
    print(f"Actual: {len(w)} warnings")

    if len(w) == 0:
        print("BUG: No deprecation warning emitted!")
    else:
        for warning in w:
            print(f"  Warning: {warning.category.__name__}: {warning.message}")

print("\nTest 3: Check if NBitBase is in module __dict__")
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")

print("\nTest 4: Check what NBitBase actually is")
print(f"type(npt.NBitBase): {type(npt.NBitBase)}")
print(f"npt.NBitBase: {npt.NBitBase}")