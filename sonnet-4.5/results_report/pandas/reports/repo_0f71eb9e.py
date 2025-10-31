import pandas.tseries.frequencies as freq_module

source, target = 'D', 'B'

is_sub_dt = freq_module.is_subperiod(source, target)
is_super_dt = freq_module.is_superperiod(source, target)
is_sub_td = freq_module.is_subperiod(target, source)
is_super_td = freq_module.is_superperiod(target, source)

print(f"is_subperiod('D', 'B') = {is_sub_dt}")
print(f"is_superperiod('D', 'B') = {is_super_dt}")
print(f"is_subperiod('B', 'D') = {is_sub_td}")
print(f"is_superperiod('B', 'D') = {is_super_td}")

assert not (is_super_dt and is_super_td), "Both cannot be superperiods of each other!"