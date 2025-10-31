from scipy.optimize import bisect, ridder, brenth, brentq

def test_method(method_name, method):
    print(f"\n{'='*50}")
    print(f"Testing {method_name}")
    print('='*50)
    
    call_count = 0

    def f(x):
        nonlocal call_count
        call_count += 1
        result = x - 1.5
        print(f"  Call {call_count}: f({x:.6f}) = {result:.6f}")
        return result

    a, b = 0.0, 2.0
    
    print(f"Initial interval: [{a}, {b}]")
    
    # Run the method
    try:
        root, result = method(f, a, b, full_output=True, disp=False)
        print(f"\nRoot found: {root}")
        print(f"Actual function calls: {call_count}")
        print(f"Reported function_calls: {result.function_calls}")
        print(f"Discrepancy: {call_count - result.function_calls}")
        
        if call_count != result.function_calls:
            print(f"⚠️  MISMATCH: Actual={call_count}, Reported={result.function_calls}")
        else:
            print("✓ Counts match")
    except Exception as e:
        print(f"Error: {e}")

# Test all bracketing methods
test_method("bisect", bisect)
test_method("ridder", ridder)
test_method("brenth", brenth)
test_method("brentq", brentq)
