import numpy.f2py.symbolic as symbolic

# Test case 1: Unmatched opening bracket causes RecursionError
print("Test 1: Calling replace_parenthesis('[') with unmatched opening bracket")
try:
    result = symbolic.replace_parenthesis('[')
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test case 2: After RecursionError, COUNTER is corrupted
print("\nTest 2: Calling replace_parenthesis('(a)') after the RecursionError")
try:
    result = symbolic.replace_parenthesis('(a)')
    print(f"Result: {result}")
except StopIteration as e:
    print(f"StopIteration occurred - COUNTER generator is corrupted")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")