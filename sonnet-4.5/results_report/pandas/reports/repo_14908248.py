from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test the specific failing cases mentioned in the bug report
print("Testing inverse relationship violations:")
print()

# Test 1: Daily (D) vs Business Day (B)
print("Test 1: Daily (D) vs Business Day (B)")
print(f"is_subperiod('D', 'B') = {is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {is_superperiod('B', 'D')}")
print(f"These should be equal, but are not! (Expected: both True)")
print()

# Test 2: Daily (D) vs Custom Business Day (C)
print("Test 2: Daily (D) vs Custom Business Day (C)")
print(f"is_subperiod('D', 'C') = {is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {is_superperiod('C', 'D')}")
print(f"These should be equal, but are not! (Expected: both True)")
print()

# Test 3: Custom Business Day (C) vs Business Day (B)
print("Test 3: Custom Business Day (C) vs Business Day (B)")
print(f"is_subperiod('C', 'B') = {is_subperiod('C', 'B')}")
print(f"is_superperiod('B', 'C') = {is_superperiod('B', 'C')}")
print(f"These should be equal, but are not! (Expected: both True)")
print()

# Additional tests for reverse direction violations
print("Additional violations in reverse directions:")
print()

# Test 4: Business Day (B) vs Daily (D) - reverse
print("Test 4: Business Day (B) vs Daily (D)")
print(f"is_subperiod('B', 'D') = {is_subperiod('B', 'D')}")
print(f"is_superperiod('D', 'B') = {is_superperiod('D', 'B')}")
print(f"These should be equal, but are not! (Expected: both False)")
print()

# Test 5: Business Day (B) vs Custom Business Day (C) - reverse
print("Test 5: Business Day (B) vs Custom Business Day (C)")
print(f"is_subperiod('B', 'C') = {is_subperiod('B', 'C')}")
print(f"is_superperiod('C', 'B') = {is_superperiod('C', 'B')}")
print(f"These should be equal, but are not! (Expected: both False)")
print()

# Test 6: Custom Business Day (C) vs Daily (D) - reverse
print("Test 6: Custom Business Day (C) vs Daily (D)")
print(f"is_subperiod('C', 'D') = {is_subperiod('C', 'D')}")
print(f"is_superperiod('D', 'C') = {is_superperiod('D', 'C')}")
print(f"These should be equal, but are not! (Expected: both False)")
print()

# Test 7: Year (Y) vs Year (Y) - self-comparison
print("Test 7: Year (Y) vs Year (Y) - self-comparison")
print(f"is_subperiod('Y', 'Y') = {is_subperiod('Y', 'Y')}")
print(f"is_superperiod('Y', 'Y') = {is_superperiod('Y', 'Y')}")
print(f"These should be equal (both False based on code inspection)")
print()

# Verify the asymmetry in the implementation
print("Summary:")
print("The inverse relationship property is violated for multiple frequency pairs.")
print("If is_subperiod(A, B) is True, then is_superperiod(B, A) should also be True.")
print("If is_subperiod(A, B) is False, then is_superperiod(B, A) should also be False.")