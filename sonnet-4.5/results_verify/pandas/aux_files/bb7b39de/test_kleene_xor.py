import numpy as np
import pandas as pd
from pandas.core.ops import kleene_xor
from hypothesis import given, strategies as st, settings

# Test 1: Property-based test from the bug report
@given(
    st.lists(st.booleans(), min_size=1, max_size=20),
    st.lists(st.booleans(), min_size=1, max_size=20),
)
@settings(max_examples=500)
def test_kleene_xor_self_is_false(values, mask_vals):
    n = max(len(values), len(mask_vals))
    left = np.array(values[:n] + [False] * (n - len(values)))
    left_mask = np.array(mask_vals[:n] + [False] * (n - len(mask_vals)))

    result, mask = kleene_xor(left, left, left_mask, left_mask)

    assert np.all(~result), f"x ^ x should be False everywhere, got {result}"
    assert np.all(~mask), f"x ^ x should not be masked (even for NA), got mask={mask}"

# Test 2: Direct reproduction from the bug report
def test_direct_reproduction():
    print("\n=== Test 2: Direct reproduction ===")
    left = np.array([False])
    left_mask = np.array([True])

    result, mask = kleene_xor(left, left, left_mask, left_mask)

    print(f"NA ^ NA returns: result={result[0]}, mask={mask[0]}")
    print(f"Expected (according to bug report): result=False, mask=False")

    # Check what the actual behavior is
    if mask[0]:
        print("ACTUAL: NA ^ NA returns NA (masked)")
    else:
        print("ACTUAL: NA ^ NA returns False (unmasked)")

# Test 3: High-level demonstration with pandas Series
def test_pandas_series():
    print("\n=== Test 3: High-level pandas Series test ===")
    s = pd.Series([True, False, pd.NA], dtype="boolean")
    result = s ^ s

    print(f"Series: {s.to_list()}")
    print(f"s ^ s:  {result.to_list()}")
    print(f"Expected (according to bug report): [False, False, False]")

# Test 4: Test different XOR combinations
def test_xor_combinations():
    print("\n=== Test 4: Testing various XOR combinations ===")

    # Test True ^ True
    left = np.array([True])
    left_mask = np.array([False])
    result, mask = kleene_xor(left, left, left_mask, left_mask)
    print(f"True ^ True: result={result[0]}, mask={mask[0]} (expected: False, unmasked)")

    # Test False ^ False
    left = np.array([False])
    left_mask = np.array([False])
    result, mask = kleene_xor(left, left, left_mask, left_mask)
    print(f"False ^ False: result={result[0]}, mask={mask[0]} (expected: False, unmasked)")

    # Test NA ^ NA (the contested case)
    left = np.array([True])  # Value doesn't matter when masked
    left_mask = np.array([True])  # Masked = NA
    result, mask = kleene_xor(left, left, left_mask, left_mask)
    print(f"NA ^ NA: result={result[0]}, mask={mask[0]} (bug report expects: False, unmasked)")

    # Test True ^ NA
    left = np.array([True])
    left_mask = np.array([False])
    right = np.array([True])
    right_mask = np.array([True])
    result, mask = kleene_xor(left, right, left_mask, right_mask)
    print(f"True ^ NA: result={result[0]}, mask={mask[0]} (expected: NA/masked)")

    # Test False ^ NA
    left = np.array([False])
    left_mask = np.array([False])
    right = np.array([False])
    right_mask = np.array([True])
    result, mask = kleene_xor(left, right, left_mask, right_mask)
    print(f"False ^ NA: result={result[0]}, mask={mask[0]} (expected: NA/masked)")

# Test 5: Test the idempotent property for different values
def test_idempotent_property():
    print("\n=== Test 5: Testing idempotent property x ^ x = False ===")

    values = [True, False, True, False, True]
    masks = [False, False, True, True, False]

    left = np.array(values)
    left_mask = np.array(masks)

    result, mask = kleene_xor(left, left, left_mask, left_mask)

    print(f"Values: {values}")
    print(f"Masks:  {masks}")
    print(f"Result: {result}")
    print(f"Result mask: {mask}")
    print(f"All results False? {np.all(~result)}")
    print(f"All unmasked (according to bug)? {np.all(~mask)}")

if __name__ == "__main__":
    # Run the direct tests
    test_direct_reproduction()
    test_pandas_series()
    test_xor_combinations()
    test_idempotent_property()

    # Run the hypothesis test with a specific failing case
    print("\n=== Test 1: Hypothesis test with specific failing case ===")
    try:
        # Call the test function directly with the specific values
        values = [False]
        mask_vals = [True]
        n = max(len(values), len(mask_vals))
        left = np.array(values[:n] + [False] * (n - len(values)))
        left_mask = np.array(mask_vals[:n] + [False] * (n - len(mask_vals)))

        result, mask = kleene_xor(left, left, left_mask, left_mask)

        assert np.all(~result), f"x ^ x should be False everywhere, got {result}"
        assert np.all(~mask), f"x ^ x should not be masked (even for NA), got mask={mask}"
        print("Hypothesis test with values=[False], mask_vals=[True]: PASSED")
    except AssertionError as e:
        print(f"Hypothesis test with values=[False], mask_vals=[True]: FAILED")
        print(f"Error: {e}")