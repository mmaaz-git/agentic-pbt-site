import numpy as np
from numpy.polynomial import polynomial as poly

# Test case from bug report
dividend = [1.0, 1.0]
divisor = [1.0, 1e-100]

print("=" * 60)
print("TESTING: dividend=[1.0, 1.0], divisor=[1.0, 1e-100]")
print("=" * 60)

quo, rem = poly.polydiv(dividend, divisor)
reconstructed = poly.polyadd(poly.polymul(quo, divisor), rem)

print(f"Dividend:      {dividend}")
print(f"Divisor:       {divisor}")
print(f"Quotient:      {quo}")
print(f"Remainder:     {rem}")
print(f"Reconstructed: {reconstructed}")

print("\n--- Checking reconstruction property ---")
print(f"Original dividend: {dividend}")
print(f"Reconstructed:     {list(reconstructed)}")

# Check if they're equal
try:
    assert np.allclose(dividend, reconstructed), \
        "polydiv violates: dividend = quotient * divisor + remainder"
    print("✓ Reconstruction property holds with np.allclose")
except AssertionError as e:
    print(f"✗ {e}")

# Let's check manually
print("\n--- Manual verification ---")
# dividend = quotient * divisor + remainder
# [1.0, 1.0] = quo * [1.0, 1e-100] + rem

# Manually calculate quotient * divisor
manual_product = poly.polymul(quo, divisor)
print(f"quotient * divisor = {manual_product}")

# Add remainder
manual_reconstruction = poly.polyadd(manual_product, rem)
print(f"+ remainder = {manual_reconstruction}")

print("\n--- Checking with different tolerances ---")
for rtol in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
    try:
        if np.allclose(dividend, reconstructed, rtol=rtol):
            print(f"✓ Matches with rtol={rtol}")
        else:
            print(f"✗ Does not match with rtol={rtol}")
    except:
        print(f"✗ Error with rtol={rtol}")

# Let's also do manual polynomial division to understand what's happening
print("\n--- Understanding the math ---")
print("Dividing (1 + x) by (1 + 1e-100*x)")
print("Using long division:")
print("The leading coefficient of divisor is 1e-100")
print("So we scale by 1/1e-100 = 1e100")

# Let's also test with a slightly larger coefficient
print("\n" + "=" * 60)
print("TESTING: dividend=[1.0, 1.0], divisor=[1.0, 1e-10]")
print("=" * 60)
dividend2 = [1.0, 1.0]
divisor2 = [1.0, 1e-10]

quo2, rem2 = poly.polydiv(dividend2, divisor2)
reconstructed2 = poly.polyadd(poly.polymul(quo2, divisor2), rem2)

print(f"Dividend:      {dividend2}")
print(f"Divisor:       {divisor2}")
print(f"Quotient:      {quo2}")
print(f"Remainder:     {rem2}")
print(f"Reconstructed: {reconstructed2}")

try:
    assert np.allclose(dividend2, reconstructed2), \
        "polydiv violates: dividend = quotient * divisor + remainder"
    print("✓ Reconstruction property holds")
except AssertionError as e:
    print(f"✗ {e}")