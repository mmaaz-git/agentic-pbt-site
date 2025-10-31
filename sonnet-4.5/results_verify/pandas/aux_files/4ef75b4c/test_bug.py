import pandas as pd
import pandas.core.ops as ops

print("Series.div exists:", hasattr(pd.Series, 'div'))
print("DataFrame.div exists:", hasattr(pd.DataFrame, 'div'))

print("\nDataFrame.div is listed in DataFrame docs:")
print("'div' in pd.DataFrame.div.__doc__:", 'div' in pd.DataFrame.div.__doc__)

print("\nmake_flex_doc supports all other flex methods:")
for method in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow']:
    try:
        result = ops.make_flex_doc(method, 'series')
        print(f"  {method}: ✓")
    except KeyError as e:
        print(f"  {method}: ✗ KeyError: {e}")

print("\nmake_flex_doc fails for 'div':")
try:
    result = ops.make_flex_doc('div', 'series')
    print("  div: ✓")
except KeyError as e:
    print(f"  div: ✗ KeyError: {e}")