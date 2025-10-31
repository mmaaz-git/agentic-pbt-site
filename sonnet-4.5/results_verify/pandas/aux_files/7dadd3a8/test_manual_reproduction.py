from pandas.compat.numpy.function import (
    ARGSORT_DEFAULTS,
    ARGSORT_DEFAULTS_KIND,
    validate_argsort_with_ascending,
)

print("ARGSORT_DEFAULTS includes 'kind':", 'kind' in ARGSORT_DEFAULTS)
print("ARGSORT_DEFAULTS_KIND includes 'kind':", 'kind' in ARGSORT_DEFAULTS_KIND)

# Also print the actual contents
print("\nARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)
print("ARGSORT_DEFAULTS_KIND:", ARGSORT_DEFAULTS_KIND)

ascending = True
kwargs = {"kind": None}

try:
    result = validate_argsort_with_ascending(ascending, (), kwargs)
    print(f"\nSuccess: {result}")
except TypeError as e:
    print(f"\nBug: {e}")