import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as hnp

# First, let's test with the simple reproduction case
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
non_contiguous = arr[::2, ::2]

print("=== Simple Reproduction Test ===")
print(f"Input contiguous: {non_contiguous.flags.contiguous}")
print(f"Input C-contiguous: {non_contiguous.flags.c_contiguous}")
print(f"Input F-contiguous: {non_contiguous.flags.f_contiguous}")

m = np.matrix(non_contiguous, copy=False)

print(f"Matrix contiguous: {m.flags.contiguous}")
print(f"Matrix C-contiguous: {m.flags.c_contiguous}")
print(f"Matrix F-contiguous: {m.flags.f_contiguous}")

# Now let's run the hypothesis test
@given(hnp.arrays(dtype=np.float64, shape=(4, 4)))
def test_dead_code_bug_noncontiguous_copy(arr):
    non_contiguous = arr[::2, ::2]

    if not non_contiguous.flags.contiguous:
        m = np.matrix(non_contiguous, copy=False)

        if not m.flags.c_contiguous and not m.flags.f_contiguous:
            assert False, "Matrix from non-contiguous array is non-contiguous"

print("\n=== Running Hypothesis Test ===")
try:
    test_dead_code_bug_noncontiguous_copy()
    print("Hypothesis test passed - no assertion failure found")
except AssertionError as e:
    print(f"Hypothesis test failed with assertion: {e}")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")