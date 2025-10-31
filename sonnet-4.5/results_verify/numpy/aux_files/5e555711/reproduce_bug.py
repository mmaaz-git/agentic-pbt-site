import warnings
import numpy.typing as npt

print("Testing deprecation warning for NBitBase:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = npt.NBitBase

    if len(w) == 0:
        print("BUG: No deprecation warning triggered")
    else:
        print(f"OK: Warning triggered - {w[0].message}")

# Also check if NBitBase is accessible
print(f"\nNBitBase is accessible: {result}")
print(f"NBitBase type: {type(result)}")