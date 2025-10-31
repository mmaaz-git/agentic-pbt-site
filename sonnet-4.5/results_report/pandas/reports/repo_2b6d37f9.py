from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq = 'Y-JAN'

super_result = is_superperiod(freq, freq)
sub_result = is_subperiod(freq, freq)

print(f"is_superperiod('{freq}', '{freq}') = {super_result}")
print(f"is_subperiod('{freq}', '{freq}') = {sub_result}")

assert super_result == sub_result, \
    f"Expected both to return the same value, but got {super_result} and {sub_result}"