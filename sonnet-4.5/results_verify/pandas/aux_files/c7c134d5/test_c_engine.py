import pandas as pd
from io import StringIO

print("Testing with C engine vs Python engine")
print("="*60)

# Test with C engine
print("\nC engine tests:")
print("-"*40)

# Test 1: || separator with C engine (should fail because C engine doesn't support multi-char)
print("Test 1: '||' separator with C engine")
try:
    csv_data = 'col0||col1\n0||1'
    df = pd.read_csv(StringIO(csv_data), sep='||', engine='c')
    print(f"Result: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 2: Single | separator with C engine (should work)
print("\nTest 2: '|' separator with C engine")
try:
    csv_data = 'col0|col1\n0|1'
    df = pd.read_csv(StringIO(csv_data), sep='|', engine='c')
    print(f"Result: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Checking documentation behavior:")
print("-"*40)

# Test what pandas says about multi-character separators
print("\nTesting multi-character separator '::' with Python engine:")
csv_data = 'col0::col1::col2\n0::1::2'
df = pd.read_csv(StringIO(csv_data), sep='::', engine='python')
print(f"Input: {repr(csv_data)}")
print(f"Result: {df.shape[1]} columns {list(df.columns)}")
print(f"DataFrame:\n{df}")

print("\nTesting escaped multi-character separator '\\|\\|' with Python engine:")
csv_data = 'col0||col1||col2\n0||1||2'
df = pd.read_csv(StringIO(csv_data), sep=r'\|\|', engine='python')
print(f"Input: {repr(csv_data)}")
print(f"Result: {df.shape[1]} columns {list(df.columns)}")
print(f"DataFrame:\n{df}")