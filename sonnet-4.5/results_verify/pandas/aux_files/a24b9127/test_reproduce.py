from pandas.tseries.frequencies import is_subperiod, is_superperiod

sub = is_subperiod('Y', 'Y')
sup = is_superperiod('Y', 'Y')

print(f"is_subperiod('Y', 'Y') = {sub}")
print(f"is_superperiod('Y', 'Y') = {sup}")
print(f"Are they equal? {sub == sup}")
assert sub == sup