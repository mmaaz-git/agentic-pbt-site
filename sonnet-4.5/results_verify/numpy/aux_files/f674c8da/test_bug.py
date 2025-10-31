from numpy.polynomial import Polynomial
import numpy as np

print("Test 1: Simple instance method call")
p = Polynomial([1, 2, 3])
print(f"Created polynomial: {p}")

try:
    p_cast = p.cast(Polynomial)
    print(f"Cast successful: {p_cast}")
except AttributeError as e:
    print(f"AttributeError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

print("Test 2: Using cast as classmethod (correct usage)")
try:
    p2 = Polynomial([1, 2, 3])
    p2_cast = Polynomial.cast(p2, Polynomial)
    print(f"Classmethod cast successful: {p2_cast}")
except Exception as e:
    print(f"Error with classmethod: {type(e).__name__}: {e}")