import pandas.io.formats.format as fmt
import numpy as np

# Test the specific failing input
result = fmt.format_percentiles([0.625, 5e-324])
print("Result for [0.625, 5e-324]:", result)

# Let's also test some edge cases
print("\n--- Testing various edge cases ---")

test_cases = [
    [0.5],
    [0.0, 1.0],
    [0.25, 0.5, 0.75],
    [0.625, 5e-324],  # The reported failing case
    [1e-10, 0.5],
    [1e-100, 0.5],
    [5e-324, 0.625],  # Reversed order
    [0.0, 5e-324, 1.0],
]

for test_case in test_cases:
    try:
        result = fmt.format_percentiles(test_case)
        print(f"Input: {test_case}")
        print(f"Output: {result}")

        # Check if any result contains 'nan'
        has_nan = any('nan' in str(r).lower() for r in result)
        if has_nan:
            print("⚠️  Contains 'nan' - This is problematic!")
        print()
    except Exception as e:
        print(f"Input: {test_case}")
        print(f"Error: {e}")
        print()