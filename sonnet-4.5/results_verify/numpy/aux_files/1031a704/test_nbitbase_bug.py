import warnings
import numpy.typing as npt

# Test 1: Simple reproduction test
print("Test 1: Simple reproduction")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj = npt.NBitBase
    print(f"Warnings caught: {len(w)}")
    print(f"Expected: 1 DeprecationWarning")
    if len(w) > 0:
        for warning in w:
            print(f"  Warning: {warning.category.__name__}: {warning.message}")

print("\n" + "="*50 + "\n")

# Test 2: Hypothesis test
print("Test 2: Hypothesis test")
def test_nbitbase_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = npt.NBitBase
        print(f"Warnings caught in hypothesis test: {len(w)}")
        if len(w) > 0:
            for warning in w:
                print(f"  Warning: {warning.category.__name__}: {warning.message}")
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning), f"Expected DeprecationWarning, got {w[0].category}"
        assert "NBitBase" in str(w[0].message), f"Expected 'NBitBase' in message, got: {w[0].message}"
        print("  Hypothesis test passed!")

try:
    test_nbitbase_deprecation()
except AssertionError as e:
    print(f"  Hypothesis test FAILED: {e}")

print("\n" + "="*50 + "\n")

# Test 3: Check what NBitBase actually is
print("Test 3: NBitBase object inspection")
print(f"Type of npt.NBitBase: {type(npt.NBitBase)}")
print(f"Module of NBitBase: {npt.NBitBase.__module__ if hasattr(npt.NBitBase, '__module__') else 'N/A'}")
print(f"Is NBitBase in npt.__all__? {'NBitBase' in npt.__all__ if hasattr(npt, '__all__') else 'No __all__'}")
print(f"Is NBitBase in dir(npt)? {'NBitBase' in dir(npt)}")