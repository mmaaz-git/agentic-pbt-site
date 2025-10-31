import pandas as pd

# Test 1: Simple reproduction case from bug report
print("Test 1: Simple reproduction with surrogate character")
try:
    df = pd.DataFrame({
        'int_col': [0],
        'float_col': [0.0],
        'str_col': ['\ud800']
    })

    print(f"DataFrame created successfully: {df}")
    print(f"String column value repr: {repr(df['str_col'][0])}")

    interchange_obj = df.__dataframe__()
    print("Interchange object created")

    result = pd.api.interchange.from_dataframe(interchange_obj)
    print(f"SUCCESS: Conversion worked! Result: {result}")

except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Test with various surrogate characters
print("Test 2: Testing various surrogate characters")
surrogate_chars = ['\ud800', '\udfff', '\ud834\udd1e']  # Low surrogate, high surrogate, valid surrogate pair

for char in surrogate_chars:
    print(f"\nTesting with character: {repr(char)}")
    try:
        df = pd.DataFrame({
            'str_col': [char]
        })
        interchange_obj = df.__dataframe__()
        result = pd.api.interchange.from_dataframe(interchange_obj)
        print(f"SUCCESS: Worked for {repr(char)}")
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError for {repr(char)}: {e}")
    except Exception as e:
        print(f"Other error for {repr(char)}: {type(e).__name__}: {e}")