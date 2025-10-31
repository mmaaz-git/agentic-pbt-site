import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')

df = pd.DataFrame({
    'a': [1.0, 2.0],
    'b': [3.0, 4.0],
    'c': [5.0, 6.0]
})

print("Testing scatter_matrix with invalid diagonal values...")

# Test 1: diagonal='invalid'
try:
    result = pandas.plotting.scatter_matrix(df, diagonal='invalid')
    print(f"✗ Accepted invalid diagonal='invalid' - BUG CONFIRMED")
except (ValueError, KeyError) as e:
    print(f"✓ Rejected diagonal='invalid' with error: {e}")

# Test 2: diagonal='0'
try:
    result = pandas.plotting.scatter_matrix(df, diagonal='0')
    print(f"✗ Accepted invalid diagonal='0' - BUG CONFIRMED")
except (ValueError, KeyError) as e:
    print(f"✓ Rejected diagonal='0' with error: {e}")

# Test 3: diagonal='foobar'
try:
    result = pandas.plotting.scatter_matrix(df, diagonal='foobar')
    print(f"✗ Accepted invalid diagonal='foobar' - BUG CONFIRMED")
except (ValueError, KeyError) as e:
    print(f"✓ Rejected diagonal='foobar' with error: {e}")

# Test 4: Valid values should still work
print("\nTesting valid diagonal values...")
try:
    result = pandas.plotting.scatter_matrix(df, diagonal='hist')
    print(f"✓ Accepted valid diagonal='hist'")
except Exception as e:
    print(f"✗ Rejected valid diagonal='hist' with error: {e}")

try:
    result = pandas.plotting.scatter_matrix(df, diagonal='kde')
    print(f"✓ Accepted valid diagonal='kde'")
except Exception as e:
    print(f"✗ Rejected valid diagonal='kde' with error: {e}")