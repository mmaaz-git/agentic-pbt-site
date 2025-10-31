#!/usr/bin/env python3

import math

def manual_cosine_similarity(a, b):
    """Manually calculate cosine similarity to understand the math"""
    print(f"\nCalculating cosine similarity for:")
    print(f"  a = {a}")
    print(f"  b = {b}")

    # Dot product
    dot_product = sum(x * y for x, y in zip(a, b))
    print(f"  Dot product: {dot_product}")

    # Magnitude of a
    sum_squares_a = sum(x * x for x in a)
    magnitude_a = sum_squares_a ** 0.5
    print(f"  Magnitude of a: sqrt({sum_squares_a}) = {magnitude_a}")

    # Magnitude of b
    sum_squares_b = sum(x * x for x in b)
    magnitude_b = sum_squares_b ** 0.5
    print(f"  Magnitude of b: sqrt({sum_squares_b}) = {magnitude_b}")

    # Final calculation
    denominator = magnitude_a * magnitude_b
    print(f"  Denominator: {magnitude_a} * {magnitude_b} = {denominator}")

    if denominator == 0:
        print(f"  Result: UNDEFINED (division by zero)")
        return None
    else:
        result = dot_product / denominator
        print(f"  Result: {dot_product} / {denominator} = {result}")
        return result

# Test cases from the bug report
test_cases = [
    ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),  # First zero
    ([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),  # Second zero
    ([0.0, 0.0], [0.0, 0.0]),            # Both zero
    ([1.0, 2.0], [3.0, 4.0]),            # Normal case
]

print("Manual Cosine Similarity Calculations")
print("=" * 50)

for a, b in test_cases:
    manual_cosine_similarity(a, b)
    print()

# Check mathematical properties
print("\n" + "=" * 50)
print("Mathematical Facts about Cosine Similarity:")
print("- Cosine similarity measures the cosine of the angle between two vectors")
print("- Formula: cos(θ) = (A·B) / (||A|| * ||B||)")
print("- When either vector has zero magnitude, the angle is undefined")
print("- Zero vectors have no direction, so measuring angle makes no sense")
print("- Common conventions for handling zero vectors:")
print("  1. Return 0 (treat as orthogonal/unrelated)")
print("  2. Return NaN (undefined)")
print("  3. Return None (undefined)")
print("  4. Raise an exception (invalid input)")
print("- No single 'correct' answer - depends on application context")