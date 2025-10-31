import pandas as pd
import inspect

split_sig = inspect.signature(pd.core.strings.accessor.StringMethods.split)
rsplit_sig = inspect.signature(pd.core.strings.accessor.StringMethods.rsplit)

print("split() signature:")
print(split_sig)
print("\nrsplit() signature:")
print(rsplit_sig)