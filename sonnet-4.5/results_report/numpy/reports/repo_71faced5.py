import pandas.tseries.frequencies

print("Example 1: Yearly to Yearly")
print(f"is_subperiod('Y', 'Y') = {pandas.tseries.frequencies.is_subperiod('Y', 'Y')}")
print(f"is_superperiod('Y', 'Y') = {pandas.tseries.frequencies.is_superperiod('Y', 'Y')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('Y', 'Y')} != {pandas.tseries.frequencies.is_superperiod('Y', 'Y')}")

print("\nExample 2: Daily and Business Day")
print(f"is_subperiod('D', 'B') = {pandas.tseries.frequencies.is_subperiod('D', 'B')}")
print(f"is_superperiod('B', 'D') = {pandas.tseries.frequencies.is_superperiod('B', 'D')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('D', 'B')} != {pandas.tseries.frequencies.is_superperiod('B', 'D')}")

print("\nExample 3: Business Day and Daily")
print(f"is_subperiod('B', 'D') = {pandas.tseries.frequencies.is_subperiod('B', 'D')}")
print(f"is_superperiod('D', 'B') = {pandas.tseries.frequencies.is_superperiod('D', 'B')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('B', 'D')} != {pandas.tseries.frequencies.is_superperiod('D', 'B')}")

print("\nExample 4: Daily and Custom Business Day")
print(f"is_subperiod('D', 'C') = {pandas.tseries.frequencies.is_subperiod('D', 'C')}")
print(f"is_superperiod('C', 'D') = {pandas.tseries.frequencies.is_superperiod('C', 'D')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('D', 'C')} != {pandas.tseries.frequencies.is_superperiod('C', 'D')}")

print("\nExample 5: Business Day and Custom Business Day")
print(f"is_subperiod('B', 'C') = {pandas.tseries.frequencies.is_subperiod('B', 'C')}")
print(f"is_superperiod('C', 'B') = {pandas.tseries.frequencies.is_superperiod('C', 'B')}")
print(f"Inverse property violated: {pandas.tseries.frequencies.is_subperiod('B', 'C')} != {pandas.tseries.frequencies.is_superperiod('C', 'B')}")