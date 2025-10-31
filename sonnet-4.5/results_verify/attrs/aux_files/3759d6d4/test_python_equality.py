#!/usr/bin/env python3
"""Test Python's equality behavior in membership testing"""

print("Testing Python's equality in membership testing:")
print("=" * 50)

# Test if 1.0 is considered equal to 1 in a tuple
test_tuple = (True, "true", "t", "yes", "y", "on", "1", 1)
print(f"Test tuple: {test_tuple}")
print(f"1.0 in test_tuple: {1.0 in test_tuple}")
print(f"0.0 in test_tuple: {0.0 in test_tuple}")
print(f"2.0 in test_tuple: {2.0 in test_tuple}")

print("\n" + "=" * 50)
# Test with False tuple
false_tuple = (False, "false", "f", "no", "n", "off", "0", 0)
print(f"False tuple: {false_tuple}")
print(f"0.0 in false_tuple: {0.0 in false_tuple}")
print(f"1.0 in false_tuple: {1.0 in false_tuple}")

print("\n" + "=" * 50)
print("Direct equality tests:")
print(f"1.0 == 1: {1.0 == 1}")
print(f"0.0 == 0: {0.0 == 0}")
print(f"True == 1: {True == 1}")
print(f"True == 1.0: {True == 1.0}")
print(f"False == 0: {False == 0}")
print(f"False == 0.0: {False == 0.0}")

print("\n" + "=" * 50)
print("Testing membership with various float values:")
for val in [1.0, 0.0, 1.5, 2.0, 0.5, -1.0]:
    in_true = val in test_tuple
    in_false = val in false_tuple
    print(f"{val:5} -> in True tuple: {in_true:5}, in False tuple: {in_false:5}")