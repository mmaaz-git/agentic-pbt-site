import pandas as pd
import io

print("=" * 60)
print("TEST: Can dtype='str' preserve the original values?")
print("=" * 60)

test_values = ['Inf', 'NaN', 'nan', 'NA', 'null', 'infinity']

for val in test_values:
    df = pd.DataFrame({'col': [val]})
    csv_str = df.to_csv(index=False, quoting=1)

    # Try reading with dtype='str'
    df_result = pd.read_csv(io.StringIO(csv_str), dtype='str')

    original = df['col'].iloc[0]
    result = df_result['col'].iloc[0]

    print(f"Value: {val:10} -> Original: {repr(original):10} -> With dtype='str': {repr(result):10} -> Match: {original == result}")

print()
print("=" * 60)
print("TEST: Check if keep_default_na=False helps")
print("=" * 60)

for val in test_values:
    df = pd.DataFrame({'col': [val]})
    csv_str = df.to_csv(index=False, quoting=1)

    # Try reading with keep_default_na=False
    df_result = pd.read_csv(io.StringIO(csv_str), keep_default_na=False)

    original = df['col'].iloc[0]
    result = df_result['col'].iloc[0]

    print(f"Value: {val:10} -> Original: {repr(original):10} -> With keep_default_na=False: {repr(result):20} -> Match: {original == result}")

print()
print("=" * 60)
print("TEST: Combination of keep_default_na=False and na_filter=False")
print("=" * 60)

for val in test_values:
    df = pd.DataFrame({'col': [val]})
    csv_str = df.to_csv(index=False, quoting=1)

    # Try reading with both parameters
    df_result = pd.read_csv(io.StringIO(csv_str), keep_default_na=False, na_filter=False)

    original = df['col'].iloc[0]
    result = df_result['col'].iloc[0]

    print(f"Value: {val:10} -> Original: {repr(original):10} -> Result: {repr(result):20} -> Match: {original == result}")