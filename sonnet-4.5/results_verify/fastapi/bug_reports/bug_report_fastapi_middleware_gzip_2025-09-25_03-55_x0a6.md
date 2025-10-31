# Bug Report: fastapi.middleware.gzip Invalid Compression Level Acceptance

**Target**: `fastapi.middleware.gzip.GZipMiddleware`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`GZipMiddleware` accepts invalid `compresslevel` values in `__init__`, but these values cause crashes when the middleware is actually invoked, leading to delayed error detection.

## Property-Based Test

```python
import gzip
import io
from hypothesis import given, strategies as st
from fastapi.middleware.gzip import GZipMiddleware


class DummyApp:
    pass


@given(st.integers())
def test_gzip_compresslevel_accepts_only_valid_values(compresslevel):
    try:
        buf = io.BytesIO()
        gzip_file = gzip.GzipFile(mode="wb", fileobj=buf, compresslevel=compresslevel)
        gzip_file.close()
        is_valid_for_gzip = True
    except (ValueError, OSError, OverflowError) as e:
        is_valid_for_gzip = False

    try:
        middleware = GZipMiddleware(DummyApp(), compresslevel=compresslevel)
        middleware_accepted = True
    except Exception:
        middleware_accepted = False

    if not is_valid_for_gzip and middleware_accepted:
        raise AssertionError(
            f"GZipMiddleware accepted invalid compresslevel={compresslevel}, "
            f"but gzip.GzipFile rejects it"
        )
```

**Failing input**: `compresslevel=-2` (also fails for `compresslevel=10`, `100`, etc.)

## Reproducing the Bug

```python
from fastapi.middleware.gzip import GZipMiddleware
import io
import gzip


class DummyApp:
    pass


middleware = GZipMiddleware(DummyApp(), compresslevel=-2)
print(f"Middleware created with compresslevel={middleware.compresslevel}")

buf = io.BytesIO()
gzip_file = gzip.GzipFile(mode="wb", fileobj=buf, compresslevel=-2)
```

Output:
```
Middleware created with compresslevel=-2
Traceback (most recent call last):
  ...
ValueError: Invalid initialization option
```

## Why This Is A Bug

The `GZipMiddleware.__init__` accepts any integer value for `compresslevel` without validation. However, the underlying `gzip.GzipFile` (used by `GZipResponder`) only accepts values in the range `[-1, 9]`. Values outside this range cause `ValueError` or `OverflowError` when the middleware is invoked to handle a request. This delayed error makes debugging harder and can cause production failures.

## Fix

```diff
--- a/fastapi/middleware/gzip.py
+++ b/fastapi/middleware/gzip.py
@@ -1,6 +1,8 @@
 class GZipMiddleware:
     def __init__(self, app: ASGIApp, minimum_size: int = 500, compresslevel: int = 9) -> None:
+        if compresslevel < -1 or compresslevel > 9:
+            raise ValueError(f"compresslevel must be between -1 and 9, got {compresslevel}")
         self.app = app
         self.minimum_size = minimum_size
         self.compresslevel = compresslevel
```