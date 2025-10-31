#!/usr/bin/env python3
"""Test the reported bug in scipy.spatial.distance.cosine"""

from scipy.spatial.distance import cosine
from hypothesis import given, strategies as st, assume
import numpy as np
import math

print("Testing scipy.spatial.distance.cosine with zero vectors")
print("="*60)

# Test 1: The exact reported case
print("\n1. Testing the exact reported case:")
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
result = cosine(u, v)
print(f"cosine([0, 0, 0], [0, 0, 0]) = {result}")
print(f"Is result NaN? {math.isnan(result)}")
print(f"Is result 0.0? {result == 0.0}")

# Test 2: One zero vector, one non-zero
print("\n2. Testing one zero vector, one non-zero:")
u = np.array([0.0, 0.0, 0.0])
v = np.array([1.0, 2.0, 3.0])
result = cosine(u, v)
print(f"cosine([0, 0, 0], [1, 2, 3]) = {result}")
print(f"Is result NaN? {math.isnan(result)}")

# Test 3: Non-zero identical vectors
print("\n3. Testing non-zero identical vectors:")
u = np.array([1.0, 2.0, 3.0])
v = np.array([1.0, 2.0, 3.0])
result = cosine(u, v)
print(f"cosine([1, 2, 3], [1, 2, 3]) = {result}")
print(f"Is result 0.0? {result == 0.0 or abs(result) < 1e-10}")

# Test 4: Run the hypothesis test
print("\n4. Running the hypothesis test from the bug report:")

@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=20),
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=20)
)
def test_cosine_distance_range(u, v):
    assume(len(u) == len(v))
    u_arr = np.array(u)
    v_arr = np.array(v)
    d = cosine(u_arr, v_arr)
    assert 0.0 <= d <= 2.0 + 1e-9, f"Distance {d} out of range for u={u}, v={v}"

try:
    test_cosine_distance_range()
    print("Hypothesis test completed without errors")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")

# Test 5: Manual calculation to understand the math
print("\n5. Manual calculation for zero vectors:")
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
dot_product = np.dot(u, v)
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)
print(f"dot(u, v) = {dot_product}")
print(f"||u|| = {norm_u}")
print(f"||v|| = {norm_v}")
print(f"Product of norms = {norm_u * norm_v}")
if norm_u * norm_v == 0:
    print("Division by zero would occur!")
    print("Cosine similarity = dot(u,v) / (||u|| * ||v||) = 0/0 = undefined")
    print("Cosine distance = 1 - cosine_similarity = 1 - undefined = undefined")

# Test 6: Check what the assertion expects
print("\n6. Checking the bug report's assertion:")
u = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
result = cosine(u, v)
try:
    assert result == 0.0, f"Expected 0.0 for identical vectors, got {result}"
    print("Assertion passed: result is 0.0")
except AssertionError as e:
    print(f"Assertion failed: {e}")
    print(f"The function returns {result}, not 0.0 as the bug report expects")