from scipy.optimize import bisect

# Simple test case
call_count = [0]

def f(x):
    call_count[0] += 1
    return x - 1.5

a, b = 0.0, 2.0

# First evaluate f(a) and f(b) to check sign condition
fa = f(a)
fb = f(b)
print(f"Initial evaluations: f({a}) = {fa}, f({b}) = {fb}")
print(f"Sign condition satisfied: {fa * fb < 0}")
print(f"Function calls after initial check: {call_count[0]}")

# Reset counter and run bisect
call_count[0] = 0
root, result = bisect(f, a, b, full_output=True, disp=False)

print(f"\nRoot found: {root}")
print(f"Actual function calls during bisect: {call_count[0]}")
print(f"Reported function_calls: {result.function_calls}")
print(f"Difference: {call_count[0] - result.function_calls}")

# Verify the bug exists
assert call_count[0] == result.function_calls, \
    f"bisect: reported {result.function_calls} calls but actually made {call_count[0]}"