from scipy.optimize import bisect

call_count = 0

def f(x):
    global call_count
    call_count += 1
    result = x - 1.5
    print(f"Call {call_count}: f({x:.6f}) = {result:.6f}")
    return result

a, b = 0.0, 2.0

# Manually check the bracketing condition first
print("Manual pre-check:")
fa = f(a)
fb = f(b)
print(f"f(a) * f(b) = {fa} * {fb} = {fa * fb}")
print(f"Calls after manual check: {call_count}\n")

# Now reset and call bisect
print("Now calling bisect with fresh counter:")
call_count = 0
root, result = bisect(f, a, b, full_output=True, disp=False)

print(f"\nRoot found: {root}")
print(f"Actual function calls during bisect: {call_count}")
print(f"Reported function_calls: {result.function_calls}")

# Check if bisect internally evaluates f(a) and f(b) again
if call_count > result.function_calls:
    print(f"⚠️  BUG CONFIRMED: bisect made {call_count} calls but reports {result.function_calls}")
elif call_count == result.function_calls:
    print("✓ Counts match - no bug")
