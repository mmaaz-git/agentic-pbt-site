import pandas.tseries.frequencies as freq

# Testing the antisymmetric property violation
result1 = freq.is_superperiod('D', 'B')
result2 = freq.is_superperiod('B', 'D')

print(f"is_superperiod('D', 'B') = {result1}")
print(f"is_superperiod('B', 'D') = {result2}")

if result1 and result2:
    print("BUG CONFIRMED: Both return True (violates antisymmetry)")
else:
    print("Antisymmetry property is satisfied")

print("\n--- Testing inverse relationship ---")
# Testing the inverse relationship between is_superperiod and is_subperiod
result3 = freq.is_subperiod('B', 'D')
result4 = freq.is_subperiod('D', 'B')

print(f"is_subperiod('B', 'D') = {result3}")
print(f"is_subperiod('D', 'B') = {result4}")

print("\n--- Expected behavior ---")
print("If is_superperiod('D', 'B') is True, then:")
print("  - is_superperiod('B', 'D') should be False (antisymmetry)")
print("  - is_subperiod('B', 'D') should be True (inverse relationship)")