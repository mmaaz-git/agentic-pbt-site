from numpy.polynomial import Polynomial

# Create a simple polynomial
p = Polynomial([1, 2, 3])

# Try to use cast() as an instance method
# (This should work since classmethods can be called on instances in Python)
print("Calling p.cast(Polynomial)...")
p_cast = p.cast(Polynomial)
print("Success:", p_cast)