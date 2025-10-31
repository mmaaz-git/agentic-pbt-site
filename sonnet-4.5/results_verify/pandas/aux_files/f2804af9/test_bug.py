import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt
import traceback

print("Testing parallel_coordinates with no numeric columns")
print("="*60)

# Test case 1: DataFrame with only class column
print("\n1. DataFrame with only class column:")
df = pd.DataFrame({'class': ['a', 'b', 'c']})
fig, ax = plt.subplots()

try:
    result = pandas.plotting.parallel_coordinates(df, 'class')
    print(f"SUCCESS: Result returned: {result}")
except IndexError as e:
    print(f"IndexError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Other Error ({type(e).__name__}): {e}")
    traceback.print_exc()
finally:
    plt.close(fig)

# Test case 2: DataFrame with empty cols list
print("\n2. DataFrame with numeric column but empty cols list:")
df2 = pd.DataFrame({'class': ['a', 'b', 'c'], 'value': [1, 2, 3]})
fig, ax = plt.subplots()

try:
    result = pandas.plotting.parallel_coordinates(df2, 'class', cols=[])
    print(f"SUCCESS: Result returned: {result}")
except IndexError as e:
    print(f"IndexError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Other Error ({type(e).__name__}): {e}")
    traceback.print_exc()
finally:
    plt.close(fig)

# Test case 3: Normal case with numeric columns
print("\n3. Normal case with numeric columns:")
df3 = pd.DataFrame({'class': ['a', 'b', 'c'], 'value1': [1, 2, 3], 'value2': [4, 5, 6]})
fig, ax = plt.subplots()

try:
    result = pandas.plotting.parallel_coordinates(df3, 'class')
    print(f"SUCCESS: Result returned: {result}")
except Exception as e:
    print(f"Error ({type(e).__name__}): {e}")
    traceback.print_exc()
finally:
    plt.close(fig)

print("\n" + "="*60)
print("Test complete")