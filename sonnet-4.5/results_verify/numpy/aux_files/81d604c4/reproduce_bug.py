import warnings
import numpy.typing as npt

print("Test 1: Accessing npt.NBitBase directly")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase

    print(f"Warnings captured: {len(w)}")
    if w:
        for warning in w:
            print(f"Warning: {warning.message}")
    else:
        print("BUG CONFIRMED: No deprecation warning was raised!")

print("\nTest 2: Manual __getattr__ call (shows warning IS defined):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__("NBitBase")
    print(f"Warnings captured: {len(w)}")
    if w:
        print(f"Warning: {w[0].message}")

print("\nTest 3: Check if NBitBase is in module globals")
print(f"NBitBase in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"NBitBase in dir(npt): {'NBitBase' in dir(npt)}")

print("\nTest 4: Try accessing with getattr function")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = getattr(npt, "NBitBase")
    print(f"Warnings captured via getattr: {len(w)}")
    if w:
        print(f"Warning: {w[0].message}")