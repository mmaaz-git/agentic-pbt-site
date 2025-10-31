import numpy as np

print("Test numpy.polydiv with same inputs")
print("=" * 50)

# Test 1: Leading zero coefficient
print("Test 1: signal=[1, 2, 3], divisor=[0, 1]")
signal_arr = [1, 2, 3]
divisor = [0, 1]
try:
    quotient, remainder = np.polydiv(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Control case
print("\nTest 2: signal=[1, 2, 3], divisor=[1, 0]")
signal_arr = [1, 2, 3]
divisor = [1, 0]
try:
    quotient, remainder = np.polydiv(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Same as scipy but with polynomials
print("\nTest 3: Polynomial representation")
print("Polynomial x^2 + 2x + 3 divided by 1 (represented as [0, 1])")
signal_arr = [1, 2, 3]  # x^2 + 2x + 3
divisor = [0, 1]  # 0*x + 1 = 1
try:
    quotient, remainder = np.polydiv(signal_arr, divisor)
    print(f"Success: quotient={quotient}, remainder={remainder}")
    print(f"This means: ({quotient[0]}x^2 + {quotient[1]}x + {quotient[2]}) * 1 + {remainder} = x^2 + 2x + 3")
except Exception as e:
    print(f"Error: {e}")

# Test with Poly1d objects
print("\nTest 4: Using poly1d objects")
p1 = np.poly1d([1, 2, 3])
p2 = np.poly1d([0, 1])
print(f"p1 = {p1}")
print(f"p2 = {p2}")
try:
    q, r = np.polydiv(p1, p2)
    print(f"Quotient: {q}")
    print(f"Remainder: {r}")
except Exception as e:
    print(f"Error: {e}")