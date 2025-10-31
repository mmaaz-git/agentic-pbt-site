from pandas.tseries import frequencies

print("Testing inverse relationship between is_subperiod and is_superperiod:")
print("=" * 60)

# Test 1: Check the inverse relationship
print("\nTest 1: Inverse Relationship (should be equal):")
is_sub_d_b = frequencies.is_subperiod('D', 'B')
is_super_b_d = frequencies.is_superperiod('B', 'D')
print(f"is_subperiod('D', 'B') = {is_sub_d_b}")
print(f"is_superperiod('B', 'D') = {is_super_b_d}")
print(f"Are they equal? {is_sub_d_b == is_super_b_d}")

# This assertion should pass but doesn't
try:
    assert is_sub_d_b == is_super_b_d
    print("✓ Inverse relationship holds")
except AssertionError:
    print("✗ FAILED: Inverse relationship violated!")

print("\n" + "=" * 60)

# Test 2: Check both superperiods (logically impossible)
print("\nTest 2: Both Superperiods (at most one should be True):")
is_super_d_b = frequencies.is_superperiod('D', 'B')
is_super_b_d_2 = frequencies.is_superperiod('B', 'D')
print(f"is_superperiod('D', 'B') = {is_super_d_b}")
print(f"is_superperiod('B', 'D') = {is_super_b_d_2}")

# Both cannot be True (circular relationship)
try:
    assert not (is_super_d_b and is_super_b_d_2)
    print("✓ No circular superperiod relationship")
except AssertionError:
    print("✗ FAILED: Circular superperiod relationship exists!")

print("\n" + "=" * 60)

# Additional context
print("\nAdditional Context:")
print(f"is_subperiod('B', 'D') = {frequencies.is_subperiod('B', 'D')}")
print(f"is_subperiod('D', 'B') = {frequencies.is_subperiod('D', 'B')}")

print("\nSummary:")
print("- 'D' (calendar day) represents all 7 days per week")
print("- 'B' (business day) represents only 5 days per week (Mon-Fri)")
print("- Therefore, 'D' is more frequent than 'B'")
print("- Expected: is_subperiod('D', 'B') should be True (can downsample)")
print("- Expected: is_superperiod('D', 'B') should be False")