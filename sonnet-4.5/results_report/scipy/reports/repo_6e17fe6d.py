from scipy.io.arff._arffread import DateAttribute

# Test with an invalid date pattern that contains no valid date components
invalid_pattern = 'date "abc"'

try:
    result_pattern, result_unit = DateAttribute._get_date_format(invalid_pattern)
    print(f"BUG: Invalid pattern '{invalid_pattern}' was accepted")
    print(f"Returned: pattern='{result_pattern}', unit='{result_unit}'")
    print(f"Expected: Should raise ValueError")
except ValueError as e:
    print(f"Correct: ValueError raised - {e}")