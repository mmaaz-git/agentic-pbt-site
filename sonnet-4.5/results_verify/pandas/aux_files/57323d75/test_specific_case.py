import pandas as pd
from io import StringIO

print("Testing the specific failing case from bug report:")
print("="*50)

rows = [(0, 2147483648)]
csv_string = "a,b\n" + "\n".join(f"{a},{b}" for a, b in rows)

try:
    result = pd.read_csv(StringIO(csv_string), dtype={'a': 'int64', 'b': 'int32'})
    print("No error raised!")
    print(f"Column 'a' dtype: {result['a'].dtype}")
    print(f"Column 'b' dtype: {result['b'].dtype}")
    print(f"Column 'a' value: {result['a'].iloc[0]}")
    print(f"Column 'b' value: {result['b'].iloc[0]} (input was 2147483648)")

    if result['b'].iloc[0] == -2147483648:
        print("\nBUG CONFIRMED: Value 2147483648 wrapped to -2147483648")

except OverflowError as e:
    print(f"OverflowError raised: {e}")
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")

print("\n" + "="*50)
print("Testing edge cases around int32 boundaries:")

test_values = [
    2147483647,  # int32_max
    2147483648,  # int32_max + 1
    2147483649,  # int32_max + 2
    -2147483648,  # int32_min
    -2147483649,  # int32_min - 1
]

for val in test_values:
    csv = f"value\n{val}"
    try:
        result = pd.read_csv(StringIO(csv), dtype={'value': 'int32'})
        print(f"Input: {val:12} -> Output: {result['value'].iloc[0]:12}")
    except OverflowError:
        print(f"Input: {val:12} -> OverflowError")