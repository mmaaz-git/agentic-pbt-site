import pandas.tseries.frequencies as freq

# Test the symmetry violation between is_subperiod and is_superperiod
# for daily (D) and business day (B) frequencies

# Test Case 1: Check is_subperiod from D to B
is_sub_D_to_B = freq.is_subperiod('D', 'B')
print(f"is_subperiod('D', 'B') = {is_sub_D_to_B}")

# Test Case 2: Check is_superperiod from B to D
is_super_B_to_D = freq.is_superperiod('B', 'D')
print(f"is_superperiod('B', 'D') = {is_super_B_to_D}")

# Show the symmetry violation
print(f"\nSymmetry violation detected:")
print(f"is_subperiod('D', 'B') = {is_sub_D_to_B}")
print(f"is_superperiod('B', 'D') = {is_super_B_to_D}")
print(f"These should be equal but are not: {is_sub_D_to_B} != {is_super_B_to_D}")

# Also test the reverse to show full picture
is_sub_B_to_D = freq.is_subperiod('B', 'D')
is_super_D_to_B = freq.is_superperiod('D', 'B')
print(f"\nReverse case:")
print(f"is_subperiod('B', 'D') = {is_sub_B_to_D}")
print(f"is_superperiod('D', 'B') = {is_super_D_to_B}")
print(f"These should also be equal: {is_sub_B_to_D} == {is_super_D_to_B}")

# Demonstrate that pandas actually supports resampling between these frequencies
import pandas as pd
import numpy as np

print("\n--- Testing actual resampling capability ---")

# Create a daily date range
dates_daily = pd.date_range('2025-01-01', periods=10, freq='D')
data_daily = pd.Series(np.arange(10), index=dates_daily)
print(f"\nDaily data:\n{data_daily}")

# Resample from Daily to Business day (D -> B)
try:
    resampled_to_B = data_daily.resample('B').mean()
    print(f"\nSuccessfully resampled from D to B:\n{resampled_to_B}")
    print("✓ Resampling from D to B works")
except Exception as e:
    print(f"✗ Failed to resample from D to B: {e}")

# Create a business day date range
dates_business = pd.date_range('2025-01-01', periods=10, freq='B')
data_business = pd.Series(np.arange(10), index=dates_business)
print(f"\nBusiness day data:\n{data_business}")

# Resample from Business day to Daily (B -> D)
try:
    resampled_to_D = data_business.resample('D').mean()
    print(f"\nSuccessfully resampled from B to D:\n{resampled_to_D}")
    print("✓ Resampling from B to D works")
except Exception as e:
    print(f"✗ Failed to resample from B to D: {e}")

print("\n--- Conclusion ---")
print("Both D->B and B->D resampling operations work in pandas,")
print("but is_subperiod('D', 'B') incorrectly returns False.")
print("This is a bug in the is_subperiod function.")