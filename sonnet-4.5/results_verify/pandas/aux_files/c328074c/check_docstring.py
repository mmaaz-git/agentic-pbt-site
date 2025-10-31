import pandas as pd

# Get the full docstring for pd.cut
print("PANDAS.CUT DOCUMENTATION")
print("=" * 80)
print(pd.cut.__doc__)
print("\n" + "=" * 80)

# Also check what IntervalArray._validate says
from pandas.core.arrays.interval import IntervalArray
print("\nIntervalArray._validate DOCUMENTATION")
print("=" * 80)
if hasattr(IntervalArray, '_validate'):
    if IntervalArray._validate.__doc__:
        print(IntervalArray._validate.__doc__)
    else:
        print("No docstring for _validate method")
        # Let's check the source
        import inspect
        print("\nSource code:")
        try:
            print(inspect.getsource(IntervalArray._validate))
        except:
            print("Could not get source")
print("=" * 80)