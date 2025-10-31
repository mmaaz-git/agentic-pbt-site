import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr

# Test first example: negative number
print("Testing negative number: -1.938987928904224e-24")
result = normalise_float_repr("-1.938987928904224e-24")
print(f"Result: {result!r}")

try:
    float_val = float(result)
    print(f"Parsed back to float: {float_val}")
except ValueError as e:
    print(f"ERROR: Cannot parse back to float: {e}")

print()

# Test second example: positive number with scientific notation
print("Testing positive number: 1.67660926681519e-08")
result = normalise_float_repr("1.67660926681519e-08")
print(f"Result: {result!r}")

try:
    float_val = float(result)
    print(f"Parsed back to float: {float_val}")
    original = float("1.67660926681519e-08")
    print(f"Original value: {original}")
    print(f"Values match: {float_val == original}")
except ValueError as e:
    print(f"ERROR: Cannot parse back to float: {e}")

print()

# Test a simple positive number for comparison
print("Testing simple positive: 123.456")
result = normalise_float_repr("123.456")
print(f"Result: {result!r}")

# Test a simple negative number
print("\nTesting simple negative: -123.456")
result = normalise_float_repr("-123.456")
print(f"Result: {result!r}")