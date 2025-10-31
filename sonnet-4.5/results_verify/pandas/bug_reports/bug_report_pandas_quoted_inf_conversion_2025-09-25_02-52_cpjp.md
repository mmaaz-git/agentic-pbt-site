# Bug Report: pandas.read_csv Converts Quoted 'Inf' Strings to Float

**Target**: `pandas.io.parsers.read_csv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When reading CSV files, `read_csv` incorrectly converts quoted string values like `"Inf"`, `"NaN"`, and `"infinity"` to their numeric equivalents (float infinity/NaN), even when they are properly quoted and should be treated as literal strings. This violates the round-trip property and the CSV quoting convention that quoted values should be treated as strings.

## Property-Based Test

```python
import pandas as pd
import io
from hypothesis import given, strategies as st, settings

@given(
    encoding=st.sampled_from(['utf-8']),
    text=st.text(min_size=1, max_size=20)
)
@settings(max_examples=100)
def test_roundtrip_with_quoting(encoding, text):
    df = pd.DataFrame({'col': [text]})
    csv_str = df.to_csv(index=False, quoting=1)
    df_result = pd.read_csv(io.StringIO(csv_str))

    assert df['col'].iloc[0] == df_result['col'].iloc[0]
```

**Failing input**: `text='Inf'` (also 'NaN', 'nan', 'NA', 'null', 'infinity', etc.)

## Reproducing the Bug

```python
import pandas as pd
import io

df = pd.DataFrame({'col': ['Inf']})
csv_str = df.to_csv(index=False, quoting=1)

print(f"CSV with QUOTE_ALL: {repr(csv_str)}")

df_result = pd.read_csv(io.StringIO(csv_str))

print(f"Original: dtype={df['col'].dtype}, value={repr(df['col'].iloc[0])}")
print(f"After:    dtype={df_result['col'].dtype}, value={repr(df_result['col'].iloc[0])}")
```

Output:
```
CSV with QUOTE_ALL: '"col"\n"Inf"\n'
Original: dtype=object, value='Inf'
After:    dtype=float64, value=np.float64(inf)
```

## Why This Is A Bug

This violates fundamental CSV parsing conventions and the round-trip property:

1. **Quoting convention violated**: In CSV format, quoted values should be treated as literal strings. When a value is quoted as `"Inf"`, it should be read as the string `'Inf'`, not converted to numeric infinity.

2. **Round-trip property broken**: A DataFrame with string column containing `'Inf'` cannot be round-tripped through CSV, even when using explicit quoting:
   ```python
   str('Inf') -> to_csv(quoting=1) -> "Inf" -> read_csv -> float(inf)
   ```

3. **Data corruption**: Users who have legitimate string data containing 'Inf', 'NaN', etc. (e.g., abbreviations, codes, or text data) will have their data silently corrupted to numeric types.

4. **Inconsistent with CSV standards**: Most CSV parsers respect quoting and only perform type inference on unquoted values.

This affects real-world usage where:
- String columns contain values that happen to match special numeric representations
- Users rely on quoting to preserve exact string values
- Data integrity requires preserving the distinction between string 'Inf' and numeric infinity

## Fix

The type inference logic in `read_csv` should respect CSV quoting:

1. **Quoted values should not undergo numeric inference**: When a field is quoted in the CSV, it should be treated as a string, not subjected to numeric type inference.

2. **Only unquoted values should be checked for special numeric values**: Values like `Inf`, `NaN`, etc. should only be converted to their numeric equivalents when they appear unquoted in the CSV.

The fix should be in the type inference logic to check whether a value was quoted in the original CSV before applying numeric conversions. This is a standard feature in most CSV parsers.