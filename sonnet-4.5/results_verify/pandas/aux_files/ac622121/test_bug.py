import pandas as pd
import io
import numpy as np

print("=" * 60)
print("TEST 1: Simple reproduction with 'Inf'")
print("=" * 60)

df = pd.DataFrame({'col': ['Inf']})
csv_str = df.to_csv(index=False, quoting=1)  # quoting=1 means QUOTE_ALL

print(f"Original DataFrame:")
print(f"  dtype: {df['col'].dtype}")
print(f"  value: {repr(df['col'].iloc[0])}")
print(f"  type:  {type(df['col'].iloc[0])}")
print()
print(f"CSV string with QUOTE_ALL: {repr(csv_str)}")
print()

df_result = pd.read_csv(io.StringIO(csv_str))

print(f"After read_csv:")
print(f"  dtype: {df_result['col'].dtype}")
print(f"  value: {repr(df_result['col'].iloc[0])}")
print(f"  type:  {type(df_result['col'].iloc[0])}")

print()
print("=" * 60)
print("TEST 2: Test with various special values")
print("=" * 60)

test_values = ['Inf', 'inf', 'NaN', 'nan', 'NA', 'null', 'infinity', '-inf', 'None']
for val in test_values:
    df = pd.DataFrame({'col': [val]})
    csv_str = df.to_csv(index=False, quoting=1)
    df_result = pd.read_csv(io.StringIO(csv_str))

    original = df['col'].iloc[0]
    result = df_result['col'].iloc[0]

    csv_value = csv_str.split('\n')[1] if len(csv_str.split('\n')) > 1 else ''
    print(f"Value: {val:10} -> CSV: {csv_value:15} -> Result: {repr(result):20} (type: {type(result).__name__})")
    if original != result and not (pd.isna(original) and pd.isna(result)):
        print(f"  *** CHANGED: {repr(original)} -> {repr(result)}")

print()
print("=" * 60)
print("TEST 3: Test without quoting")
print("=" * 60)

# Test without quoting to see if unquoted values are also converted
df = pd.DataFrame({'col': ['Inf']})
csv_str = df.to_csv(index=False, quoting=0)  # quoting=0 means QUOTE_MINIMAL

print(f"CSV string with QUOTE_MINIMAL: {repr(csv_str)}")
df_result = pd.read_csv(io.StringIO(csv_str))
print(f"Result: dtype={df_result['col'].dtype}, value={repr(df_result['col'].iloc[0])}")

print()
print("=" * 60)
print("TEST 4: Test with na_filter=False")
print("=" * 60)

df = pd.DataFrame({'col': ['Inf']})
csv_str = df.to_csv(index=False, quoting=1)
print(f"CSV string: {repr(csv_str)}")

df_result = pd.read_csv(io.StringIO(csv_str), na_filter=False)
print(f"With na_filter=False: dtype={df_result['col'].dtype}, value={repr(df_result['col'].iloc[0])}")

print()
print("=" * 60)
print("TEST 5: Test round-trip property with Hypothesis")
print("=" * 60)

from hypothesis import given, strategies as st, settings
import sys

failed_cases = []

@given(
    text=st.sampled_from(['Inf', 'inf', 'NaN', 'nan', 'NA', 'null', 'infinity', '-inf', 'None'])
)
@settings(max_examples=20)
def test_roundtrip_with_quoting(text):
    df = pd.DataFrame({'col': [text]})
    csv_str = df.to_csv(index=False, quoting=1)
    df_result = pd.read_csv(io.StringIO(csv_str))

    original = df['col'].iloc[0]
    result = df_result['col'].iloc[0]

    # Check if they are equal (accounting for NaN equality)
    if pd.isna(original) and pd.isna(result):
        return  # Both are NaN, consider this equal

    if original != result:
        failed_cases.append((text, original, result))
        assert False, f"Round-trip failed: {repr(original)} -> {repr(result)}"

try:
    test_roundtrip_with_quoting()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed!")
    print("Failed cases:")
    for text, orig, res in failed_cases:
        print(f"  Input: {text:10} -> Original: {repr(orig):10} -> Result: {repr(res)}")