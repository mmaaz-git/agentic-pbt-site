import pandas.io.formats.format as fmt

result = fmt.format_percentiles([0.625, 5e-324])
print(result)