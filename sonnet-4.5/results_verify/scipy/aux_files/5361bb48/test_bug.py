import numpy as np
from scipy import integrate

print("Testing newton_cotes with list input and non-uniform spacing:")
print("=" * 60)

# Test case from bug report - list with non-uniform spacing
rn_list = [0, 0.5, 2]
print(f"Input (list): {rn_list}")
print(f"equal parameter: 0")

try:
    an, Bn = integrate.newton_cotes(rn_list, equal=0)
    print(f"Success! Weights: {an}, Error coefficient: {Bn}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing with numpy array (same values):")
rn_array = np.array([0, 0.5, 2])
print(f"Input (array): {rn_array}")
print(f"equal parameter: 0")

try:
    an, Bn = integrate.newton_cotes(rn_array, equal=0)
    print(f"Success! Weights: {an}, Error coefficient: {Bn}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing with list and uniform spacing:")
rn_uniform_list = [0, 1, 2]
print(f"Input (list): {rn_uniform_list}")
print(f"equal parameter: 0")

try:
    an, Bn = integrate.newton_cotes(rn_uniform_list, equal=0)
    print(f"Success! Weights: {an}, Error coefficient: {Bn}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing with list and equal=1:")
rn_list2 = [0, 0.5, 2]
print(f"Input (list): {rn_list2}")
print(f"equal parameter: 1")

try:
    an, Bn = integrate.newton_cotes(rn_list2, equal=1)
    print(f"Success! Weights: {an}, Error coefficient: {Bn}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing with integer input (normal usage):")
print(f"Input (int): 2")
print(f"equal parameter: 0 (default)")

try:
    an, Bn = integrate.newton_cotes(2)
    print(f"Success! Weights: {an}, Error coefficient: {Bn}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")