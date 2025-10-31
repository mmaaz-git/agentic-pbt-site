import numpy as np

print("Understanding polynomial coefficient representation")
print("=" * 50)

# The key question: what does [0, 1] mean as a polynomial?
print("In NumPy's polynomial convention:")
print("[a, b, c] represents a*x^2 + b*x + c")
print("So [0, 1] represents 0*x + 1 = 1")
print()

# Create polynomials to verify
p1 = np.poly1d([0, 1])
print(f"np.poly1d([0, 1]) = {p1}")
print(f"Evaluating at x=5: {p1(5)}")
print(f"Evaluating at x=10: {p1(10)}")
print()

p2 = np.poly1d([1, 0])
print(f"np.poly1d([1, 0]) = {p2}")
print(f"Evaluating at x=5: {p2(5)}")
print(f"Evaluating at x=10: {p2(10)}")
print()

# Check coefficient stripping
p3 = np.poly1d([0, 0, 1])
print(f"np.poly1d([0, 0, 1]) = {p3}")
print(f"Coefficients: {p3.coeffs}")
print()

p4 = np.poly1d([0, 0, 0, 1])
print(f"np.poly1d([0, 0, 0, 1]) = {p4}")
print(f"Coefficients: {p4.coeffs}")
print()

# What about all zeros?
p5 = np.poly1d([0, 0, 0])
print(f"np.poly1d([0, 0, 0]) = {p5}")
print(f"Coefficients: {p5.coeffs}")
print()

# Now let's see the mathematical validity
print("Mathematical validity test:")
print("If we have signal = x^2 + 2x + 3 and divisor = 1")
print("Then quotient should be x^2 + 2x + 3 and remainder should be 0")
print()

# Using properly normalized polynomials
signal = np.poly1d([1, 2, 3])
divisor = np.poly1d([1])  # Just 1, not [0, 1]
q, r = signal / divisor, signal % divisor
print(f"Using poly1d division: {signal} / {divisor}")
print(f"Quotient: {q}")
print(f"Remainder: {r}")
print()

# Now verify the relationship for deconvolution
print("For deconvolution, we need: signal = convolve(divisor, quotient) + remainder")
print("With polynomial multiplication, this becomes: signal = divisor * quotient + remainder")