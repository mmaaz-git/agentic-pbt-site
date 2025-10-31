import numpy as np
import scipy.ndimage

print("Testing scipy.ndimage.shift with mode='wrap' invertibility")
print("=" * 60)

arr = np.array([0., 1., 2., 3., 4.])
shift_amount = 2

print(f"Original array:       {arr}")
print(f"Shift amount:         {shift_amount}")
print()

# Shift forward by 2
shifted = scipy.ndimage.shift(arr, shift_amount, order=0, mode='wrap')
print(f"After shift by {shift_amount}:    {shifted}")

# Shift back by -2
shifted_back = scipy.ndimage.shift(shifted, -shift_amount, order=0, mode='wrap')
print(f"After shift by {-shift_amount}:   {shifted_back}")

print()
print(f"Expected (original):  {arr}")
print(f"Match: {np.array_equal(arr, shifted_back)}")
print()

if not np.array_equal(arr, shifted_back):
    print("ERROR: The round-trip shift is NOT equal to the original array!")
    print(f"Difference: {shifted_back - arr}")
    print(f"Last element should be {arr[-1]} but got {shifted_back[-1]}")
else:
    print("SUCCESS: Round-trip shift equals original array")

# Let's also test with grid-wrap mode (available since SciPy 1.6)
print("\n" + "=" * 60)
print("Testing with mode='grid-wrap' (if available)")
print("=" * 60)

try:
    shifted_grid = scipy.ndimage.shift(arr, shift_amount, order=0, mode='grid-wrap')
    print(f"After shift by {shift_amount} (grid-wrap):    {shifted_grid}")

    shifted_back_grid = scipy.ndimage.shift(shifted_grid, -shift_amount, order=0, mode='grid-wrap')
    print(f"After shift by {-shift_amount} (grid-wrap):   {shifted_back_grid}")

    print(f"Expected (original):            {arr}")
    print(f"Match with grid-wrap: {np.array_equal(arr, shifted_back_grid)}")

    if not np.array_equal(arr, shifted_back_grid):
        print(f"Difference with grid-wrap: {shifted_back_grid - arr}")
except:
    print("grid-wrap mode not available or error occurred")

# Let's understand what's happening step by step
print("\n" + "=" * 60)
print("Detailed step-by-step analysis")
print("=" * 60)

# Manual calculation of what should happen with wrap mode
print(f"Original: {arr}")
print("\nWith wrap mode and shift=2, elements should rotate:")
print("Position 0 <- Position 3 (value 3)")
print("Position 1 <- Position 4 (value 4)")
print("Position 2 <- Position 0 (value 0)")
print("Position 3 <- Position 1 (value 1)")
print("Position 4 <- Position 2 (value 2)")
expected_shift = np.array([3., 4., 0., 1., 2.])
print(f"Expected after shift by 2: {expected_shift}")
print(f"Actual after shift by 2:   {shifted}")
print(f"Shift forward matches expected: {np.array_equal(shifted, expected_shift)}")

print("\nNow shifting back by -2:")
print("Position 0 <- Position 2 (from shifted: value 0)")
print("Position 1 <- Position 3 (from shifted: value 1)")
print("Position 2 <- Position 4 (from shifted: value 2)")
print("Position 3 <- Position 0 (from shifted: value 3)")
print("Position 4 <- Position 1 (from shifted: value 4)")
expected_back = np.array([0., 1., 2., 3., 4.])
print(f"Expected after shift by -2: {expected_back}")
print(f"Actual after shift by -2:   {shifted_back}")