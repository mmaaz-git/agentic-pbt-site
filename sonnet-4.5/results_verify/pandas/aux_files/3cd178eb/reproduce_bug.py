from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)
print("ARGSORT_DEFAULTS['kind']:", ARGSORT_DEFAULTS['kind'])

expected_from_line_138 = "quicksort"
actual_value = ARGSORT_DEFAULTS['kind']

print(f"\nLine 138 sets kind='{expected_from_line_138}'")
print(f"Line 140 overwrites kind={actual_value}")
print(f"Result: Line 138 has no effect (dead code)")