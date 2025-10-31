# Bug Report: Starlette FileResponse Range Merging

**Target**: `starlette.responses.FileResponse._parse_range_header`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The range merging algorithm in `FileResponse._parse_range_header` fails to properly merge overlapping HTTP Range headers, resulting in overlapping ranges in the output.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from starlette.responses import FileResponse, MalformedRangeHeader, RangeNotSatisfiable


def generate_valid_range_str(file_size):
    ranges_strategy = st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=file_size-1),
            st.integers(min_value=0, max_value=file_size-1)
        ).map(lambda r: (min(r), max(r)) if r[0] != r[1] else (r[0], r[0]+1)),
        min_size=1,
        max_size=10
    )
    return ranges_strategy.map(
        lambda ranges: "bytes=" + ",".join(f"{s}-{e-1}" for s, e in ranges)
    )


@given(st.integers(min_value=100, max_value=10000).flatmap(
    lambda fs: st.tuples(st.just(fs), generate_valid_range_str(fs))
))
@settings(max_examples=500)
def test_parsed_ranges_are_non_overlapping(args):
    file_size, http_range = args

    try:
        result = FileResponse._parse_range_header(http_range, file_size)
    except (MalformedRangeHeader, RangeNotSatisfiable):
        assume(False)
        return

    for i in range(len(result) - 1):
        start1, end1 = result[i]
        start2, end2 = result[i+1]
        assert end1 <= start2, (
            f"Ranges overlap or are not sorted: "
            f"range {i} = ({start1}, {end1}), "
            f"range {i+1} = ({start2}, {end2}), "
            f"input = {http_range}"
        )
```

**Failing input**: `http_range = "bytes=100-199,400-499,150-450"` with `file_size = 1000`

## Reproducing the Bug

```python
from starlette.responses import FileResponse

file_size = 1000
http_range = "bytes=100-199,400-499,150-450"

result = FileResponse._parse_range_header(http_range, file_size)

print(f"Result: {result}")

for i in range(len(result) - 1):
    start1, end1 = result[i]
    start2, end2 = result[i+1]
    if end1 > start2:
        print(f"BUG: Ranges overlap: ({start1}, {end1}) and ({start2}, {end2})")
```

Output:
```
Result: [(100, 451), (400, 500)]
BUG: Ranges overlap: (100, 451) and (400, 500)
```

Expected output:
```
Result: [(100, 500)]
```

## Why This Is A Bug

The HTTP Range header specification requires that range requests be properly parsed and merged. The code intends to merge overlapping ranges (as evidenced by the merging logic), but fails when:
1. A new range merges with an earlier range in the result
2. The merged range now overlaps with a later range in the result
3. The algorithm breaks after the first merge without checking subsequent ranges

This violates the invariant that the returned ranges should be non-overlapping and sorted.

## Fix

```diff
--- a/starlette/responses.py
+++ b/starlette/responses.py
@@ -490,12 +490,24 @@ class FileResponse(Response):

         # Merge ranges
         result: list[tuple[int, int]] = []
         for start, end in ranges:
-            for p in range(len(result)):
-                p_start, p_end = result[p]
-                if start > p_end:
-                    continue
-                elif end < p_start:
-                    result.insert(p, (start, end))  # THIS IS NOT REACHED!
-                    break
-                else:
-                    result[p] = (min(start, p_start), max(end, p_end))
-                    break
-            else:
-                result.append((start, end))
+            merged = False
+            for p in range(len(result)):
+                p_start, p_end = result[p]
+                if start > p_end:
+                    continue
+                elif end < p_start:
+                    result.insert(p, (start, end))
+                    merged = True
+                    break
+                else:
+                    # Merge and continue checking other ranges
+                    start = min(start, p_start)
+                    end = max(end, p_end)
+                    result.pop(p)
+                    # Don't break - continue to merge with other overlapping ranges
+
+            if not merged:
+                # Insert the (possibly merged) range in the correct sorted position
+                for p in range(len(result)):
+                    if start < result[p][0]:
+                        result.insert(p, (start, end))
+                        break
+                else:
+                    result.append((start, end))

         return result
```