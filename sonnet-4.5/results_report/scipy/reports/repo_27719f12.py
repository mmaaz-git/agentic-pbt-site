from scipy.io.arff._arffread import DateAttribute

# Test with a completely invalid pattern that has no date components
pattern = "A"
try:
    result_pattern, unit = DateAttribute._get_date_format(f"date {pattern}")
    print(f"Pattern: {pattern}")
    print(f"Result: {result_pattern}")
    print(f"Unit: {unit}")
    print("ERROR: Should have raised ValueError for invalid pattern")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")