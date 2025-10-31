from pandas.tseries import frequencies

print("Testing the relationships between 'D' (day) and 'B' (business day):")
print()

# First test - inverse relationship
is_sub_d_b = frequencies.is_subperiod('D', 'B')
is_super_b_d = frequencies.is_superperiod('B', 'D')

print(f"is_subperiod('D', 'B') = {is_sub_d_b}")
print(f"is_superperiod('B', 'D') = {is_super_b_d}")
print(f"These should be equal (inverse relationship), but they are: {is_sub_d_b == is_super_b_d}")
print()

# Second test - both superperiods
is_super_d_b = frequencies.is_superperiod('D', 'B')
is_super_b_d = frequencies.is_superperiod('B', 'D')

print(f"is_superperiod('D', 'B') = {is_super_d_b}")
print(f"is_superperiod('B', 'D') = {is_super_b_d}")
print(f"Both are True: {is_super_d_b and is_super_b_d}")
print("This is logically impossible - two frequencies cannot both be superperiods of each other!")
print()

# Let's also check the subperiod relationships
is_sub_b_d = frequencies.is_subperiod('B', 'D')
print(f"is_subperiod('B', 'D') = {is_sub_b_d}")
print(f"is_subperiod('D', 'B') = {is_sub_d_b}")

# Run the assertions to see failures
try:
    assert is_sub_d_b == is_super_b_d
    print("✓ First assertion passed (inverse relationship)")
except AssertionError:
    print("✗ First assertion failed: is_subperiod('D', 'B') != is_superperiod('B', 'D')")

try:
    assert not (is_super_d_b and is_super_b_d)
    print("✓ Second assertion passed (not both superperiods)")
except AssertionError:
    print("✗ Second assertion failed: Both is_superperiod('D', 'B') and is_superperiod('B', 'D') are True")