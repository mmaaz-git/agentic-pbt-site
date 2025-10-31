import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from scipy.optimize.cython_optimize import _zeros

# Test case 1: Should converge (error_num = 0)
print("Test 1: Converging case")
args = (0.0, 0.0, 0.0, 1.0)  # x^3
xa, xb = -1.0, 1.0
xtol, rtol, mitr = 1e-6, 1e-6, 100
output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)
print(f"  error_num: {output['error_num']} (expected 0 for convergence)")
print(f"  root: {output['root']}")
print()

# Test case 2: Non-convergence (error_num = -2)
print("Test 2: Non-converging case (limited iterations)")
args = (0.0, 0.0, 0.0, 1.0)  # x^3
xa, xb = -1.0, 2.0
xtol, rtol, mitr = 1e-6, 1e-6, 10
output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)
print(f"  error_num: {output['error_num']} (expected -2 for convergence error)")
print(f"  iterations: {output['iterations']}/{mitr}")
print()

# Test case 3: Sign error (error_num = -1) - both endpoints have same sign
print("Test 3: Attempting sign error case")
# Let's try to create a case where f(xa) and f(xb) have same sign
# For x^3, if both xa and xb are positive, they'll have same sign
args = (0.0, 0.0, 0.0, 1.0)  # x^3
xa, xb = 1.0, 2.0
xtol, rtol, mitr = 1e-6, 1e-6, 100
try:
    output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)
    print(f"  error_num: {output['error_num']}")
    print(f"  Output: {output}")
except Exception as e:
    print(f"  Exception raised: {e}")