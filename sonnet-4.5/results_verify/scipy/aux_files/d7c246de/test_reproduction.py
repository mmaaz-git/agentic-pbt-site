from scipy.optimize import bisect

call_count = 0

def f(x):
    global call_count
    call_count += 1
    print(f"Call {call_count}: f({x})")
    return x - 1.5

a, b = 0.0, 2.0
root, result = bisect(f, a, b, full_output=True, disp=False)

print(f"Actual function calls: {call_count}")
print(f"Reported function_calls: {result.function_calls}")
print(f"Difference: {call_count - result.function_calls}")
