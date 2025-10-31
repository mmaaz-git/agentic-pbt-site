import scipy.optimize.cython_optimize._zeros as zeros

# Test what polynomial the function actually evaluates
# Let's use a simple polynomial where we can easily verify

# Test case: Find root of 2x + 4 = 0, which should be x = -2
print("Test case: Finding root of 2x + 4 = 0")
print("Expected root: x = -2")
print()

# If args are in ascending order (c0, c1, c2, c3) for c0 + c1*x + c2*x^2 + c3*x^3
# Then for 2x + 4 = 0, we need: c0=4, c1=2, c2=0, c3=0
print("Trying ascending order: args = (4, 2, 0, 0)")
args = (4.0, 2.0, 0.0, 0.0)
try:
    result = zeros.full_output_example(args, -5.0, 0.0, 1e-6, 1e-6, 100)
    print(f"Root found: {result['root']}")
    print(f"Verification: 4 + 2*{result['root']} = {4 + 2*result['root']}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50 + "\n")

# If args are in descending order (c3, c2, c1, c0) for c3*x^3 + c2*x^2 + c1*x + c0
# Then for 2x + 4 = 0, we need: c3=0, c2=0, c1=2, c0=4
print("Trying descending order: args = (0, 0, 2, 4)")
args = (0.0, 0.0, 2.0, 4.0)
try:
    result = zeros.full_output_example(args, -5.0, 0.0, 1e-6, 1e-6, 100)
    print(f"Root found: {result['root']}")
    print(f"Verification (if descending): 0*x^3 + 0*x^2 + 2*x + 4 = {2*result['root'] + 4}")
    print(f"What it actually computed (ascending): 0 + 0*x + 2*x^2 + 4*x^3 = {0 + 0*result['root'] + 2*result['root']**2 + 4*result['root']**3}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50 + "\n")

# More complex test: x^2 - 4 = 0, roots should be ±2
print("Test case: Finding root of x^2 - 4 = 0")
print("Expected roots: x = ±2")
print()

# Ascending order: c0=-4, c1=0, c2=1, c3=0
print("Using ascending order: args = (-4, 0, 1, 0)")
args = (-4.0, 0.0, 1.0, 0.0)
result = zeros.full_output_example(args, 1.0, 3.0, 1e-6, 1e-6, 100)
print(f"Root found: {result['root']}")
print(f"Verification: -4 + 0*x + 1*x^2 + 0*x^3 = {-4 + 0*result['root'] + 1*result['root']**2 + 0*result['root']**3}")