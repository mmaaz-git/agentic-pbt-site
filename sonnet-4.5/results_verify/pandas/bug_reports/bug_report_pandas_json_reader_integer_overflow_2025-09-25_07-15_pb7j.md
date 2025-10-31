# Bug Report: pandas.io.json JsonReader Rejects Valid Large Negative Integers

**Target**: `pandas.io.json.JsonReader` (exposed via `pandas.api.typing.JsonReader`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`pd.read_json()` and `JsonReader` reject integers below -2^63 with "ValueError: Value is too small", while accepting integers above 2^63-1. This asymmetric handling violates JSON round-trip properties for valid JSON data.

## Property-Based Test

```python
import pandas as pd
import pandas.api.typing as pat
from hypothesis import given, strategies as st, settings
import io
import json


@settings(max_examples=30)
@given(
    st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=122)),
            st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=20)),
            min_size=1,
            max_size=5
        ),
        min_size=1,
        max_size=20
    )
)
def test_json_reader_round_trip(data_list):
    json_str = json.dumps(data_list)
    json_buffer = io.StringIO(json_str)

    reader = pat.JsonReader(
        filepath_or_buffer=json_buffer,
        orient=None,
        typ='frame',
        dtype=None,
        convert_axes=None,
        convert_dates=True,
        keep_default_dates=True,
        precise_float=False,
        date_unit=None,
        encoding=None,
        lines=False,
        chunksize=None,
        compression='infer',
        nrows=None
    )

    result = reader.read()
    reader.close()
```

**Failing input**: `[{'A': -9223372036854775809}]` (i.e., -(2^63 + 1))

## Reproducing the Bug

```python
import pandas as pd
import io
import json

data = [{"value": -9223372036854775809}]
json_str = json.dumps(data)

df = pd.read_json(io.StringIO(json_str))
```

Output:
```
ValueError: Value is too small
```

## Why This Is A Bug

1. **Asymmetric behavior**: `pd.read_json()` accepts integers > 2^63-1 but rejects integers < -2^63
2. **JSON standard violation**: JSON supports arbitrary precision integers
3. **Python compatibility**: Python's `json.loads()` handles these values correctly
4. **Round-trip failure**: Valid JSON that Python can parse cannot be loaded by pandas

Examples:
- `-9223372036854775809` (-(2^63 + 1)): **FAILS**
- `-9223372036854775808` (-(2^63)): **works**
- `9223372036854775807` (2^63 - 1): **FAILS** (in dict format)
- `9223372036854775808` (2^63): **works** (in list format)

The asymmetry means pandas can write JSON it cannot read back.

## Fix

The issue stems from ujson's integer parsing. pandas should either:

1. Fall back to Python's json module for values outside int64 range
2. Document this limitation clearly
3. Provide better error messages indicating the value and suggesting workarounds

Workaround for users:
```python
import json
data = json.loads(json_string)
df = pd.DataFrame(data)
```

A proper fix would involve modifying `pandas/io/json/_json.py` to handle large integers gracefully, potentially by detecting overflow and falling back to Python's json parser or converting to object dtype.