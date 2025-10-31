import warnings
import numpy.typing as npt

print("Testing NBitBase deprecation warning...\n")

# Test 1: Direct access to NBitBase (should show warning but doesn't)
print("Test 1: Direct access to npt.NBitBase")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase

    print(f"Warnings captured: {len(w)}")
    if len(w) == 0:
        print("BUG CONFIRMED: No deprecation warning was raised when accessing NBitBase!")
    else:
        for warning in w:
            print(f"Warning: {warning.message}")

print("\n" + "="*50 + "\n")

# Test 2: Verify the warning exists in __getattr__ (manual call)
print("Test 2: Manual __getattr__ call (to prove the warning IS defined)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__("NBitBase")

    print(f"Warnings captured: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")

print("\n" + "="*50 + "\n")

# Test 3: Show NBitBase is in module namespace
print("Test 3: Checking module namespace")
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"'NBitBase' in dir(npt): {'NBitBase' in dir(npt)}")

print("\n" + "="*50 + "\n")

# Test 4: Demonstrate the attribute lookup order issue
print("Test 4: Understanding Python's attribute lookup order")
print("When accessing npt.NBitBase, Python:")
print("1. First checks npt.__dict__ (module globals)")
print("2. Only calls __getattr__ if not found in __dict__")
print(f"\nSince NBitBase IS in __dict__, __getattr__ is never called!")
print(f"NBitBase object from direct access: {npt.NBitBase}")
print(f"NBitBase object from __getattr__: {npt.__getattr__('NBitBase')}")
print(f"They are the same object: {npt.NBitBase is npt.__getattr__('NBitBase')}")