import scipy.optimize.cython_optimize._zeros as zeros

# Test case from the bug report
args = (-2.0, 0.0, 0.0, 1.0)
result = zeros.full_output_example(args, 1.0, 2.0, 1e-9, 1e-9, 100)

print(f"Result: {result}")
print(f"Iterations: {result['iterations']}")
print(f"Max iterations: 100")
print(f"Error number: {result['error_num']}")
print()

# Let's run multiple times to see if values change
print("Running 5 times to check consistency:")
for i in range(5):
    result = zeros.full_output_example(args, 1.0, 2.0, 1e-9, 1e-9, 100)
    print(f"Run {i+1}: iterations={result['iterations']}, error_num={result['error_num']}, root={result['root']}")