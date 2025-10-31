import pandas as pd
from io import StringIO
import traceback

# Test case 1: Simple reproduction from bug report
print("Test 1: Simple reproduction with duplicate index_col=[0, 0]")
csv_data = "a,b,c\n1,2,3\n4,5,6"
try:
    df = pd.read_csv(StringIO(csv_data), index_col=[0, 0])
    print("Success - no error raised")
    print(df)
except Exception as e:
    print(f"Error raised: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test case 2: Try with string column names duplicated
print("Test 2: Duplicate string column names in index_col")
try:
    df = pd.read_csv(StringIO(csv_data), index_col=['a', 'a'])
    print("Success - no error raised")
    print(df)
except Exception as e:
    print(f"Error raised: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 3: Try with mixed duplicates
print("Test 3: Mixed duplicate (integer and string)")
try:
    df = pd.read_csv(StringIO(csv_data), index_col=[0, 'a'])
    print("Success - no error raised")
    print(df)
except Exception as e:
    print(f"Error raised: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 4: Non-duplicate index_col (should work)
print("Test 4: Non-duplicate index_col=[0, 1] (should work)")
try:
    df = pd.read_csv(StringIO(csv_data), index_col=[0, 1])
    print("Success - no error raised")
    print(df)
except Exception as e:
    print(f"Error raised: {type(e).__name__}: {e}")