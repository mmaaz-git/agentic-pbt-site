"""Test Python's built-in range behavior to understand expected semantics."""

# Test different range configurations
test_cases = [
    (1, 0, 1),    # Empty - start > stop with positive step
    (0, 1, 1),    # One element
    (0, 5, 1),    # Multiple elements
    (0, 5, 2),    # Step size > 1
    (5, 0, -1),   # Negative step
    (0, 0, 1),    # Empty - start == stop
    (-5, -10, -1), # Negative values with negative step
    (10, 5, 1),   # Empty - larger start > stop with positive step
]

print("Python's range behavior:")
print("=" * 50)

for start, stop, step in test_cases:
    r = range(start, stop, step)
    elements = list(r)
    print(f"range({start:3}, {stop:3}, {step:2}) -> len={len(r):2}, elements={elements}")

print("\n" + "=" * 50)
print("\nKey observations:")
print("1. range(1, 0, 1) has length 0 (empty range)")
print("2. range(0, 5, 2) has length 3 with elements [0, 2, 4]")
print("3. Python's len(range) always returns >= 0, never negative")
print("4. The formula for length is more complex than simple division")

# Show Python's actual calculation
print("\nPython's range length calculation (from docs):")
print("For positive step: max(0, (stop - start + step - 1) // step)")
print("For negative step: max(0, (start - stop - step - 1) // (-step))")

# Verify formula
print("\nVerifying formula:")
for start, stop, step in test_cases:
    r = range(start, stop, step)
    actual = len(r)
    if step > 0:
        calculated = max(0, (stop - start + step - 1) // step)
    else:
        calculated = max(0, (start - stop - step - 1) // (-step))
    match = "✓" if actual == calculated else "✗"
    print(f"{match} range({start:3}, {stop:3}, {step:2}): actual={actual:2}, calculated={calculated:2}")