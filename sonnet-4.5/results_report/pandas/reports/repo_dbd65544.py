import pandas.tseries.frequencies as frequencies

freqs = ['D', 'W', 'M', 'Q', 'Y', 'h', 'min', 's']

for freq in freqs:
    is_sub = frequencies.is_subperiod(freq, freq)
    is_super = frequencies.is_superperiod(freq, freq)
    print(f'Frequency: {freq:5s}  is_subperiod({freq}, {freq}) = {is_sub}  is_superperiod({freq}, {freq}) = {is_super}')