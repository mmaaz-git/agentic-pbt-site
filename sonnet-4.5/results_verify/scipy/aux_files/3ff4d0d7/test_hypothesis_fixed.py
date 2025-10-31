from scipy.optimize import bisect, ridder, brenth, brentq

def test_iterations_field_boundary_root(root, offset):
    def f(x):
        return x - root

    a = root - offset
    b = root

    for method in [bisect, ridder, brenth, brentq]:
        root_val, info = method(f, a, b, full_output=True)

        assert isinstance(info.iterations, int), \
            f"{method.__name__}: iterations is not an int, type={type(info.iterations)}"

        if not (0 <= info.iterations <= 1000):
            print(f"FAILURE: {method.__name__}: iterations = {info.iterations} (should be small non-negative int)")
            return False
    return True

# Test with specific failing input mentioned in bug report
print("Testing with specific input: root=0.0, offset=1.0")
success = test_iterations_field_boundary_root(root=0.0, offset=1.0)
if success:
    print("Test passed")
else:
    print("Test failed as expected")

print("\nTesting with another example: root=5.0, offset=5.0")
success = test_iterations_field_boundary_root(root=5.0, offset=5.0)
if success:
    print("Test passed")
else:
    print("Test failed as expected")