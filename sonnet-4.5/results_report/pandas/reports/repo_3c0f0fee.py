import pandas.io.formats.format as fmt
import warnings

warnings.simplefilter("always")
result = fmt.format_percentiles([0.5, 0.5, 0.5])
print(f"Result: {result}")