from scipy.io.arff._arffread import DateAttribute

pattern = "A"
result_pattern, unit = DateAttribute._get_date_format(f"date {pattern}")

print(f"Pattern: {pattern}")
print(f"Result: {result_pattern}")
print(f"Unit: {unit}")