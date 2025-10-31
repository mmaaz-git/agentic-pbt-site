import pandas as pd

# See what's available in pandas.api.indexers
import pandas.api.indexers as indexers
print("Available indexers in pandas.api.indexers:")
for attr in dir(indexers):
    if not attr.startswith('_'):
        print(f"  - {attr}")