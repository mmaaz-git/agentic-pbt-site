import numpy as np

def _round_frac(x, precision: int):
    """
    Round the fractional part of the given number
    """
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return np.around(x, digits)

# Test with the problematic values
values = [-1.11253693e-311, 5.56268465e-309, 1.11253693e-308]
precision = 3

print("Testing _round_frac with tiny values:")
for val in values:
    try:
        result = _round_frac(val, precision)
        print(f"  {val:.5e} -> {result}")
    except Exception as e:
        print(f"  {val:.5e} -> ERROR: {e}")

# What happens with very small numbers near zero
print("\nLog calculation for tiny positive number:")
tiny_positive = 1.11253693e-308
print(f"  Value: {tiny_positive}")
print(f"  log10(abs({tiny_positive})): {np.log10(abs(tiny_positive))}")
digits = -int(np.floor(np.log10(abs(tiny_positive)))) - 1 + 3
print(f"  Calculated digits: {digits}")
print(f"  np.around({tiny_positive}, {digits}): {np.around(tiny_positive, digits)}")

# Check what happens with the negative value
print("\nNegative tiny value:")
tiny_negative = -1.11253693e-311
print(f"  Value: {tiny_negative}")
print(f"  log10(abs({tiny_negative})): {np.log10(abs(tiny_negative))}")
try:
    digits = -int(np.floor(np.log10(abs(tiny_negative)))) - 1 + 3
    print(f"  Calculated digits: {digits}")
    print(f"  np.around({tiny_negative}, {digits}): {np.around(tiny_negative, digits)}")
except Exception as e:
    print(f"  ERROR: {e}")