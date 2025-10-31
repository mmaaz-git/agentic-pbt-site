import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr

# Test case 1: Negative scientific notation that produces invalid output
test_value1 = "-1.938987928904224e-24"
print(f"Input: {test_value1}")
result1 = normalise_float_repr(test_value1)
print(f"Output: {result1!r}")

try:
    float_result1 = float(result1)
    print(f"Float conversion successful: {float_result1}")
except ValueError as e:
    print(f"Float conversion failed: {e}")

print()

# Test case 2: Positive scientific notation that produces incorrect value
test_value2 = "1.67660926681519e-08"
print(f"Input: {test_value2}")
result2 = normalise_float_repr(test_value2)
print(f"Output: {result2!r}")

try:
    float_result2 = float(result2)
    print(f"Float conversion successful: {float_result2}")
    original2 = float(test_value2)
    print(f"Original value: {original2}")
    print(f"Values match: {float_result2 == original2}")
except ValueError as e:
    print(f"Float conversion failed: {e}")

print()

# Test case 3: Simple negative number (for comparison - this works)
test_value3 = "-123.456"
print(f"Input: {test_value3}")
result3 = normalise_float_repr(test_value3)
print(f"Output: {result3!r}")

try:
    float_result3 = float(result3)
    print(f"Float conversion successful: {float_result3}")
    original3 = float(test_value3)
    print(f"Original value: {original3}")
    print(f"Values match: {float_result3 == original3}")
except ValueError as e:
    print(f"Float conversion failed: {e}")