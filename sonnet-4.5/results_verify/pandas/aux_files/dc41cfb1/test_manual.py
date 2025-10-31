import pandas as pd
from io import StringIO

print("Testing basic case with '00' and '0':")
df = pd.DataFrame({'00': [1, 2], '0': [3, 4]})
print("Original columns:", list(df.columns))
print("Original column types:", [type(c).__name__ for c in df.columns])

json_str = df.to_json(orient='split')
print("\nJSON output:", json_str)

result = pd.read_json(StringIO(json_str), orient='split')
print("\nResult columns:", list(result.columns))
print("Result column types:", [type(c).__name__ for c in result.columns])
print("Data loss: '00' became", result.columns[0])

print("\n" + "="*50 + "\n")

# Test additional examples
test_cases = [
    (['0'], "Single numeric string '0'"),
    (['0', '1'], "Multiple numeric strings"),
    (['00'], "String with leading zero"),
    (['0', 'a'], "Mixed numeric and non-numeric"),
    (['abc'], "Non-numeric string"),
]

for cols, description in test_cases:
    print(f"Test: {description}")
    data = {col: [1] for col in cols}
    df = pd.DataFrame(data)
    print(f"  Original: {list(df.columns)}")

    json_str = df.to_json(orient='split')
    result = pd.read_json(StringIO(json_str), orient='split')
    print(f"  Result:   {list(result.columns)}")
    print(f"  Types:    {[type(c).__name__ for c in result.columns]}")
    print()

# Test with convert_axes parameter
print("Testing with convert_axes=False:")
df = pd.DataFrame({'00': [1, 2], '0': [3, 4]})
json_str = df.to_json(orient='split')
result = pd.read_json(StringIO(json_str), orient='split', convert_axes=False)
print(f"Original: {list(df.columns)}")
print(f"Result:   {list(result.columns)}")
print(f"Types:    {[type(c).__name__ for c in result.columns]}")