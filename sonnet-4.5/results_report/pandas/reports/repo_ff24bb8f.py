from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)
print(f"ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']}")

# Show that line 138 is dead code - the value 'quicksort' is never used
# because line 140 immediately overwrites it with None
assert ARGSORT_DEFAULTS["kind"] is None, "Expected kind to be None due to overwrite on line 140"
print("Assertion passed: ARGSORT_DEFAULTS['kind'] is None")
print("\nThis demonstrates that line 138 (setting kind='quicksort') is dead code")
print("because line 140 immediately overwrites it with None")