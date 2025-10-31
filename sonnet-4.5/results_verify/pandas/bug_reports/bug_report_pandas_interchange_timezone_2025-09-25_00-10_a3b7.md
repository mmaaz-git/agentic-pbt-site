# Bug Report: pandas.core.interchange.parse_datetime_format_str Invalid Timezone Handling

**Target**: `pandas.core.interchange.from_dataframe.parse_datetime_format_str`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_datetime_format_str` function crashes with unhelpful error messages when given invalid timezone strings through the DataFrame interchange protocol. This can happen when receiving data from external libraries with malformed timezone information.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st
from pandas.core.interchange.from_dataframe import parse_datetime_format_str


@given(
    resolution=st.sampled_from(['s', 'm', 'u', 'n']),
    tz=st.text(min_size=1, max_size=20)
)
def test_parse_datetime_format_str_handles_invalid_tz(resolution, tz):
    format_str = f"ts{resolution}:{tz}"
    data = np.array([0, 1000, 2000], dtype=np.int64)
    result = parse_datetime_format_str(format_str, data)
```

**Failing input**: `resolution='s', tz='0'` and `resolution='s', tz='\x80'`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.interchange.from_dataframe import parse_datetime_format_str

data = np.array([0, 1000, 2000], dtype=np.int64)

format_str = "tss:0"
result = parse_datetime_format_str(format_str, data)
```

Output:
```
pytz.exceptions.UnknownTimeZoneError: '0'
```

## Why This Is A Bug

1. The `parse_datetime_format_str` function is part of the DataFrame interchange protocol implementation, which receives format strings from external dataframe libraries
2. Invalid timezone strings (e.g., '0', '\x80', or any non-existent timezone) cause the function to crash with `pytz.exceptions.UnknownTimeZoneError`
3. This error is raised deep in the call stack (through pandas -> pytz) without helpful context about the interchange protocol
4. The function should either:
   - Validate timezone strings and raise a clear error message
   - Catch the `UnknownTimeZoneError` and re-raise with better context
   - Document that it expects valid IANA timezone names

The interchange protocol is designed to work across different dataframe libraries, so it should be robust to malformed inputs or provide clear error messages to help users debug issues.

## Fix

Add timezone validation or better error handling in the `parse_datetime_format_str` function:

```diff
diff --git a/pandas/core/interchange/from_dataframe.py b/pandas/core/interchange/from_dataframe.py
index 1234567..abcdefg 100644
--- a/pandas/core/interchange/from_dataframe.py
+++ b/pandas/core/interchange/from_dataframe.py
@@ -4,6 +4,7 @@ import re
 from typing import Any

 import numpy as np
+import pytz

 from pandas._config import using_string_dtype

@@ -372,7 +373,14 @@ def parse_datetime_format_str(format_str, data) -> pd.Series | np.ndarray:
             unit += "s"
         data = data.astype(f"datetime64[{unit}]")
         if tz != "":
-            data = pd.Series(data).dt.tz_localize("UTC").dt.tz_convert(tz)
+            try:
+                data = pd.Series(data).dt.tz_localize("UTC").dt.tz_convert(tz)
+            except pytz.exceptions.UnknownTimeZoneError as e:
+                raise ValueError(
+                    f"Invalid timezone '{tz}' in interchange protocol format string '{format_str}'. "
+                    f"Expected a valid IANA timezone name (e.g., 'UTC', 'US/Eastern'). "
+                    f"Original error: {e}"
+                ) from e
         return data

     # date 'td{Days/Ms}'
```