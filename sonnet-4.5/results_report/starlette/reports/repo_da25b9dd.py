import numpy.polynomial.polynomial as poly

# Test with empty coefficient array
result = poly.polyval(2.0, [])
print(f"Result: {result}")