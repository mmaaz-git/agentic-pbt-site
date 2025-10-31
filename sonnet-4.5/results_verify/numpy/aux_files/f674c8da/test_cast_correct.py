"""Test to find the actual correct usage of cast()"""

from numpy.polynomial import Polynomial, Chebyshev
import numpy as np

# Create a polynomial
p = Polynomial([1, 2, 3])
print(f"Original Polynomial: {p}")
print(f"Type: {type(p)}")

# Create a Chebyshev polynomial to test conversion
c = Chebyshev([1, 2, 3])
print(f"\nOriginal Chebyshev: {c}")
print(f"Type: {type(c)}")

print("\n" + "="*60)
print("Testing different ways to call cast():")

# Test 1: Convert Chebyshev to Polynomial using Polynomial.cast()
print("\n1. Polynomial.cast(chebyshev_instance):")
try:
    result = Polynomial.cast(c)
    print(f"Success! Converted Chebyshev to Polynomial: {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# Test 2: What about casting Polynomial to Polynomial?
print("\n2. Polynomial.cast(polynomial_instance):")
try:
    result = Polynomial.cast(p)
    print(f"Success! Cast Polynomial to Polynomial: {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# Test 3: Instance method style with Chebyshev
print("\n3. chebyshev_instance.cast(Polynomial) - instance converting to different type:")
try:
    result = c.cast(Polynomial)
    print(f"Success! {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# Test 4: Instance method style with Polynomial
print("\n4. polynomial_instance.cast(Polynomial) - instance converting to same type:")
try:
    result = p.cast(Polynomial)
    print(f"Success! {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

# Test 5: Using convert() instance method directly (what cast calls internally)
print("\n5. Using convert() instance method directly:")
try:
    result = p.convert(kind=Polynomial)
    print(f"Success with p.convert(kind=Polynomial): {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")

print("\n6. Converting Chebyshev to Polynomial with convert():")
try:
    result = c.convert(kind=Polynomial)
    print(f"Success with c.convert(kind=Polynomial): {result}")
    print(f"Type: {type(result)}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")