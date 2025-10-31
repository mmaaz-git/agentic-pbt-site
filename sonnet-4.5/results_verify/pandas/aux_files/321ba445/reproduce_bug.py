import pandas.tseries.frequencies as freq

print(f"is_subperiod('D', 'B') = {freq.is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {freq.is_superperiod('B', 'D')}")

print(f"is_subperiod('D', 'C') = {freq.is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {freq.is_superperiod('C', 'D')}")

print(f"is_subperiod('B', 'C') = {freq.is_subperiod('B', 'C')}")
print(f"is_superperiod('C', 'B') = {freq.is_superperiod('C', 'B')}")