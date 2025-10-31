from scipy.spatial.distance import correlation
import numpy as np

# Test various constant arrays
test_cases = [
    ([1.0, 1.0], [1.0, 1.0]),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([-5.0, -5.0, -5.0, -5.0], [-5.0, -5.0, -5.0, -5.0]),
    ([100.0] * 10, [100.0] * 10),
]

print("Testing constant arrays:")
for u, v in test_cases:
    u_arr = np.array(u)
    v_arr = np.array(v)
    result = correlation(u_arr, v_arr)
    print(f"  correlation({u[:3]}{'...' if len(u) > 3 else ''}, {v[:3]}{'...' if len(v) > 3 else ''}) = {result}")

# Test non-constant arrays
print("\nTesting non-constant arrays:")
u1 = np.array([1.0, 2.0, 3.0])
v1 = np.array([1.0, 2.0, 3.0])
print(f"  correlation([1, 2, 3], [1, 2, 3]) = {correlation(u1, v1)}")

u2 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, 5.0, 6.0])
print(f"  correlation([1, 2, 3], [4, 5, 6]) = {correlation(u2, v2)}")

# Test edge case: different constant arrays
print("\nTesting different constant arrays:")
u3 = np.array([5.0, 5.0, 5.0])
v3 = np.array([10.0, 10.0, 10.0])
result = correlation(u3, v3)
print(f"  correlation([5, 5, 5], [10, 10, 10]) = {result}")