import llm

print("Bug 1: Division by zero")
print("Testing llm.cosine_similarity([0, 0], [1, 1])...")
try:
    result = llm.cosine_similarity([0, 0], [1, 1])
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\nBug 2: Incorrect result for different lengths")
print("Testing llm.cosine_similarity([1, 2, 3], [4, 5])...")
try:
    result = llm.cosine_similarity([1, 2, 3], [4, 5])
    print(f"Result: {result}")

    # Show what's happening internally
    print("\nAnalysis of what's happening:")
    a = [1, 2, 3]
    b = [4, 5]

    # What the function does with zip (only 2 terms)
    dot_product_actual = sum(x * y for x, y in zip(a, b))
    print(f"Dot product (using zip, truncated): {dot_product_actual}")
    print(f"  Calculation: 1*4 + 2*5 = {1*4} + {2*5} = {dot_product_actual}")

    # Full magnitude calculations
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    print(f"Magnitude of a (full length): {magnitude_a}")
    print(f"  Calculation: sqrt(1² + 2² + 3²) = sqrt({1**2 + 2**2 + 3**2}) = {magnitude_a}")
    print(f"Magnitude of b (full length): {magnitude_b}")
    print(f"  Calculation: sqrt(4² + 5²) = sqrt({4**2 + 5**2}) = {magnitude_b}")

    # What the function returns
    incorrect_result = dot_product_actual / (magnitude_a * magnitude_b)
    print(f"\nIncorrect result from function: {incorrect_result}")

    # What the correct calculation would be (with zero-padding)
    dot_product_correct = 1*4 + 2*5 + 3*0
    correct_result = dot_product_correct / (magnitude_a * magnitude_b)
    print(f"Correct result (with zero-padding): {correct_result}")

except Exception as e:
    print(f"Error: {e}")