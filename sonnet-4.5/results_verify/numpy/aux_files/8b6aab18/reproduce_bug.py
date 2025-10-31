import numpy.polynomial as np_poly

print("Test 1:")
p = np_poly.Polynomial([1])
p_trunc = p.truncate(3)
print(f"Original polynomial coefficients: {p.coef}")
print(f"After truncate(3) coefficients: {p_trunc.coef}")
print(f"Length: {len(p_trunc.coef)}, Expected: 3")

print("\nTest 2:")
p2 = np_poly.Polynomial([0.0])
p2_trunc = p2.truncate(2)
print(f"Original polynomial coefficients: {p2.coef}")
print(f"After truncate(2) coefficients: {p2_trunc.coef}")
print(f"Length: {len(p2_trunc.coef)}, Expected: 2")

print("\nTest 3: Truncating to smaller size")
p3 = np_poly.Polynomial([1, 2, 3, 4, 5])
p3_trunc = p3.truncate(3)
print(f"Original polynomial coefficients: {p3.coef}")
print(f"After truncate(3) coefficients: {p3_trunc.coef}")
print(f"Length: {len(p3_trunc.coef)}, Expected: 3")