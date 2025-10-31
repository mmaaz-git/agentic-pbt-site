import pandas as pd

# Check if the function is part of the public API
public_api_check = hasattr(pd, 'core') and hasattr(pd.core, 'indexers')
print(f"pandas.core accessible: {hasattr(pd, 'core')}")
print(f"pandas.core.indexers accessible: {hasattr(pd.core, 'indexers') if hasattr(pd, 'core') else 'N/A'}")

# Check if it's meant to be internal (starts with underscore)
from pandas.core.indexers import length_of_indexer
print(f"Function name: {length_of_indexer.__name__}")
print(f"Is private (starts with _): {length_of_indexer.__name__.startswith('_')}")

# Check module docstring
import pandas.core.indexers.utils
print(f"\nModule docstring: {pandas.core.indexers.utils.__doc__}")

# Check if there's any warning about internal use
print(f"\nFunction docstring: {length_of_indexer.__doc__}")