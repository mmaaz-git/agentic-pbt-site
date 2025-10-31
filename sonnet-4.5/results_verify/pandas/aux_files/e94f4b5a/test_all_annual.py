from pandas.tseries.frequencies import is_subperiod, is_superperiod

annual_frequencies = ['Y', 'Y-JAN', 'Y-FEB', 'Y-MAR', 'Y-APR', 'Y-MAY', 'Y-JUN',
                      'Y-JUL', 'Y-AUG', 'Y-SEP', 'Y-OCT', 'Y-NOV', 'Y-DEC',
                      'YS', 'BY', 'BYS']

print("Testing annual frequencies comparing to self:")
for freq in annual_frequencies:
    super_result = is_superperiod(freq, freq)
    sub_result = is_subperiod(freq, freq)
    print(f"{freq:8} - is_superperiod: {super_result}, is_subperiod: {sub_result}")

print("\nTesting non-annual frequencies comparing to self:")
other_frequencies = ['D', 'W', 'M', 'Q', 'Q-JAN', 'h', 'min']
for freq in other_frequencies:
    super_result = is_superperiod(freq, freq)
    sub_result = is_subperiod(freq, freq)
    print(f"{freq:8} - is_superperiod: {super_result}, is_subperiod: {sub_result}")