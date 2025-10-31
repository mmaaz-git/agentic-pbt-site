from scipy.integrate import simpson
import numpy as np

y = np.array([0.0, 0.0, 0.0, 1.0])
x = np.array([0.0, 1.0, 2.0, 3.0])

forward = simpson(y, x=x)
backward = simpson(y[::-1], x=x[::-1])

print(f"Forward:  {forward}")
print(f"Backward: {backward}")
print(f"Expected: {-forward}")
print(f"Difference: {forward + backward}")

# Let's also test with odd number of points to see if the issue persists
print("\n--- Testing with odd number of points (N=5) ---")
y_odd = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
x_odd = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

forward_odd = simpson(y_odd, x=x_odd)
backward_odd = simpson(y_odd[::-1], x=x_odd[::-1])

print(f"Forward:  {forward_odd}")
print(f"Backward: {backward_odd}")
print(f"Expected: {-forward_odd}")
print(f"Difference: {forward_odd + backward_odd}")

# Let's test with another even number case
print("\n--- Testing with even number of points (N=6) ---")
y_even = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
x_even = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

forward_even = simpson(y_even, x=x_even)
backward_even = simpson(y_even[::-1], x=x_even[::-1])

print(f"Forward:  {forward_even}")
print(f"Backward: {backward_even}")
print(f"Expected: {-forward_even}")
print(f"Difference: {forward_even + backward_even}")