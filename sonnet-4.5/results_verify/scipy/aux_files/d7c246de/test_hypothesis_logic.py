from scipy.optimize import bisect

def test_bisect_funcalls():
    """Reproduce the exact logic from the hypothesis test"""
    a, b = 0.0, 2.0  # Simple test case
    
    call_count = [0]

    def f(x):
        call_count[0] += 1
        print(f"Call {call_count[0]}: f({x})")
        return x - 1.5

    # This is what the hypothesis test does
    print("Step 1: Check bracketing condition")
    fa, fb = f(a), f(b)
    print(f"fa * fb = {fa} * {fb} = {fa * fb} < 0: {fa * fb < 0}")
    print(f"Calls after bracketing check: {call_count[0]}")
    
    # Reset counter as in the hypothesis test
    print("\nStep 2: Reset counter and call bisect")
    call_count[0] = 0
    root, result = bisect(f, a, b, full_output=True, disp=False)

    actual_calls = call_count[0]
    reported_calls = result.function_calls

    print(f"\nResults:")
    print(f"Root found: {root}")
    print(f"Actual calls (after reset): {actual_calls}")
    print(f"Reported calls: {reported_calls}")
    
    if actual_calls == reported_calls:
        print("✓ Test would PASS - counts match")
    else:
        print(f"✗ Test would FAIL - mismatch of {actual_calls - reported_calls}")

test_bisect_funcalls()
