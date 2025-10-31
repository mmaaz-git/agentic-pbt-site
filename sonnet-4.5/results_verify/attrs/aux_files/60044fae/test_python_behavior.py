print("Testing Python's 'in' operator behavior with numeric types:")
print(f"0.0 == 0: {0.0 == 0}")
print(f"1.0 == 1: {1.0 == 1}")
print(f"0.0 in (0,): {0.0 in (0,)}")
print(f"1.0 in (1,): {1.0 in (1,)}")
print(f"type(0.0): {type(0.0)}")
print(f"type(0): {type(0)}")
print(f"0.0 is 0: {0.0 is 0}")

# Test the actual implementation behavior
test_tuple = (True, "true", "t", "yes", "y", "on", "1", 1)
print(f"\n1.0 in {test_tuple}: {1.0 in test_tuple}")

# Test with complex numbers (which equal 1 and 0 under certain conditions)
import sys
print(f"\ncomplex(0) == 0: {complex(0) == 0}")
print(f"complex(0) in (0,): {complex(0) in (0,)}")
print(f"complex(1) == 1: {complex(1) == 1}")
print(f"complex(1) in (1,): {complex(1) in (1,)}")