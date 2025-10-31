import pandas as pd
import dask.dataframe as dd
import traceback

print("=" * 60)
print("Detailed analysis of the bug")
print("=" * 60)

# Create test data
pdf = pd.DataFrame({'values': [1.0, 2.0, 3.0], 'other': [4.0, 5.0, 6.0]})
ddf = dd.from_pandas(pdf, npartitions=2)

print("\n1. Testing DataFrame.nlargest() alone (should work):")
try:
    result = ddf.nlargest(2, 'values')
    print(f"   Type of result: {type(result)}")
    computed = result.compute()
    print(f"   Computed successfully: {computed.values.tolist()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Testing column selection after nlargest (the bug):")
try:
    result = ddf.nlargest(2, 'values')['values']
    print(f"   Type of result before compute: {type(result)}")
    computed = result.compute()
    print(f"   Computed successfully: {computed.tolist()}")
except Exception as e:
    print(f"   Error: {e}")
    print(f"   Full traceback:")
    traceback.print_exc()

print("\n3. Testing if this is specific to single column selection:")
try:
    result = ddf.nlargest(2, 'values')[['values', 'other']]
    print(f"   Type of result: {type(result)}")
    computed = result.compute()
    print(f"   Multi-column selection works: {computed.values.tolist()}")
except Exception as e:
    print(f"   Multi-column selection error: {e}")

print("\n4. Comparing with pandas behavior:")
print(f"   pandas single column: {pdf.nlargest(2, 'values')['values'].tolist()}")
print(f"   pandas multi column: {pdf.nlargest(2, 'values')[['values', 'other']].values.tolist()}")

print("\n5. Testing Series.nlargest() directly (should work):")
try:
    result = ddf['values'].nlargest(2)
    print(f"   Type: {type(result)}")
    computed = result.compute()
    print(f"   Success: {computed.tolist()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Analysis of the issue:")
print("=" * 60)
print("""
The bug occurs when:
1. DataFrame.nlargest() is called with a column parameter
2. Followed by single column selection using []
3. This creates a Series object
4. But dask still passes 'columns' parameter to Series.nlargest()
5. Series.nlargest() doesn't accept 'columns' parameter

This is a lazy evaluation issue where the operation transformation
from DataFrame to Series doesn't properly update the method parameters.
""")