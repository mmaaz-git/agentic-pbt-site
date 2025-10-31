from pandas.plotting._matplotlib.converter import TimeFormatter

formatter = TimeFormatter(locs=[])
x = 86400.99999999997

result = formatter(x)