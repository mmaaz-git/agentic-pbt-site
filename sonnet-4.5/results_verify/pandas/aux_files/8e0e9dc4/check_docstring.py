from pandas.tseries.api import guess_datetime_format
import inspect

# Get the docstring
print("Function docstring:")
print("=" * 80)
print(guess_datetime_format.__doc__)
print("=" * 80)

# Get the signature
print("\nFunction signature:")
print(inspect.signature(guess_datetime_format))