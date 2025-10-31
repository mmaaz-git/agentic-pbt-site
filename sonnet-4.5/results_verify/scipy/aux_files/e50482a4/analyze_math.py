import numpy as np

# Reproduce the calculation for kind='rank'
data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
score = 1.0
n = len(data)

print(f"Data: {data}")
print(f"Score: {score}")
print(f"n = {n}")

# For kind='rank':
# left = count(a < score)
# right = count(a <= score)
# plus1 = left < right
# perct = (left + right + plus1) * (50.0 / n)

a = np.array(data)
left = np.sum(a < score)
right = np.sum(a <= score)
plus1 = 1 if left < right else 0

print(f"\nleft (a < score): {left}")
print(f"right (a <= score): {right}")
print(f"plus1: {plus1}")

# Calculate step by step
numerator = (left + right + plus1)
factor = 50.0 / n
result = numerator * factor

print(f"\nnumerator (left + right + plus1): {numerator}")
print(f"factor (50.0 / n): {factor}")
print(f"result: {result}")
print(f"result repr: {repr(result)}")
print(f"result > 100: {result > 100}")

# Check the exact arithmetic
print(f"\nExact calculation:")
print(f"(10 + 11 + 1) * (50.0 / 11) = 22 * (50.0 / 11)")
print(f"= 22 * 4.545454545454545...")
print(f"= 100.0")

# But with floating point:
print(f"\nFloating point calculation:")
print(f"50.0 / 11 = {50.0 / 11}")
print(f"22 * (50.0 / 11) = {22 * (50.0 / 11)}")
print(f"Difference from 100: {22 * (50.0 / 11) - 100}")

# Test with 'weak' kind as well
left_weak = 0  # count(a < score) = 0
right_weak = 11  # count(a <= score) = 11
result_weak = right_weak * (100.0 / n)
print(f"\nFor kind='weak':")
print(f"right (a <= score): {right_weak}")
print(f"result = {right_weak} * (100.0 / {n}) = {result_weak}")
print(f"result > 100: {result_weak > 100}")