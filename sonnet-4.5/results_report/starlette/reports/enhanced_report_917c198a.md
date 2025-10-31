# Bug Report: Starlette FileResponse Range Merging Fails to Merge Overlapping Ranges

**Target**: `starlette.responses.FileResponse._parse_range_header`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The range merging algorithm in `FileResponse._parse_range_header` fails to properly merge overlapping HTTP Range headers when a merge operation creates a new range that overlaps with subsequent ranges in the list, resulting in overlapping ranges in the output instead of a single merged range.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, settings
from starlette.responses import FileResponse, MalformedRangeHeader, RangeNotSatisfiable


def generate_valid_range_str(file_size):
    ranges_strategy = st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=file_size-1),
            st.integers(min_value=0, max_value=file_size-1)
        ).map(lambda r: (min(r), max(r)) if r[0] != r[1] else (r[0], r[0]+1 if r[0] < file_size - 1 else r[0])),
        min_size=1,
        max_size=10
    )
    return ranges_strategy.map(
        lambda ranges: "bytes=" + ",".join(f"{s}-{e-1 if e > s else e}" for s, e in ranges)
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


if __name__ == "__main__":
    test_parsed_ranges_are_non_overlapping()
```

<details>

<summary>
**Failing input**: `bytes=0-0,2-2,0-2` with `file_size=100`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 44, in <module>
    test_parsed_ranges_are_non_overlapping()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 20, in test_parsed_ranges_are_non_overlapping
    lambda fs: st.tuples(st.just(fs), generate_valid_range_str(fs))
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 35, in test_parsed_ranges_are_non_overlapping
    assert end1 <= start2, (
           ^^^^^^^^^^^^^^
AssertionError: Ranges overlap or are not sorted: range 0 = (0, 3), range 1 = (2, 3), input = bytes=0-0,2-2,0-2
Falsifying example: test_parsed_ranges_are_non_overlapping(
    args=(100, 'bytes=0-0,2-2,0-2'),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/51/hypo.py:36
```
</details>

## Reproducing the Bug

```python
from starlette.responses import FileResponse

file_size = 1000
http_range = "bytes=100-199,400-499,150-450"

result = FileResponse._parse_range_header(http_range, file_size)

print(f"Input: http_range = {repr(http_range)}, file_size = {file_size}")
print(f"Result: {result}")
print()

for i in range(len(result) - 1):
    start1, end1 = result[i]
    start2, end2 = result[i+1]
    if end1 > start2:
        print(f"BUG: Ranges overlap: ({start1}, {end1}) and ({start2}, {end2})")
        print(f"  Range {i} ends at {end1}, but range {i+1} starts at {start2}")
        print(f"  These ranges should have been merged into a single range")

if len(result) == 2 and result[0][0] == 100 and result[0][1] == 451 and result[1][0] == 400 and result[1][1] == 500:
    print("\nExpected behavior: These overlapping ranges should be merged into [(100, 500)]")
```

<details>

<summary>
BUG: Overlapping ranges returned instead of merged range
</summary>
```
Input: http_range = 'bytes=100-199,400-499,150-450', file_size = 1000
Result: [(100, 451), (400, 500)]

BUG: Ranges overlap: (100, 451) and (400, 500)
  Range 0 ends at 451, but range 1 starts at 400
  These ranges should have been merged into a single range

Expected behavior: These overlapping ranges should be merged into [(100, 500)]
```
</details>

## Why This Is A Bug

This violates expected behavior in several ways:

1. **Violates code intent**: The code has a comment "# Merge ranges" at line 492 and contains explicit logic to merge overlapping ranges, showing clear intent to merge.

2. **Violates RFC 7233**: The HTTP specification (RFC 7233 Section 4.1) states that servers "MAY coalesce any of the ranges that overlap" and recommends coalescing overlapping ranges for efficiency. The specification also notes security considerations about "egregious range requests" with overlapping ranges.

3. **Breaks invariant**: The function returns overlapping ranges when it should return non-overlapping, sorted ranges. This violates the expected postcondition that ranges in the output should not overlap.

4. **Algorithm flaw**: The bug occurs because when the algorithm merges a range with an earlier range in the result list (line 503), it immediately breaks without checking if the newly merged range now overlaps with subsequent ranges. The algorithm needs to continue checking and merging until no more overlaps exist.

5. **Inefficiency**: Returning overlapping ranges forces the server to send redundant data in multipart responses, wasting bandwidth and processing time.

## Relevant Context

The bug manifests when:
1. Three or more ranges are provided in the HTTP Range header
2. A later range (e.g., `150-450`) overlaps with an earlier range (e.g., `100-199`)
3. After merging, the combined range (e.g., `100-451`) now overlaps with another range (e.g., `400-500`)
4. The algorithm stops after the first merge and doesn't detect the new overlap

The comment at line 500 `# THIS IS NOT REACHED!` indicates awareness of dead code, suggesting the algorithm may have been incompletely implemented.

Code location: `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/responses.py:492-508`

## Proposed Fix

```diff
--- a/starlette/responses.py
+++ b/starlette/responses.py
@@ -491,17 +491,26 @@ class FileResponse(Response):

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
+            # Keep merging until no more overlaps
+            merged = False
+            i = 0
+            while i < len(result):
+                p_start, p_end = result[i]
+                if end < p_start:
+                    # Insert before this range
+                    result.insert(i, (start, end))
+                    merged = True
+                    break
+                elif start <= p_end:
+                    # Overlaps - merge and continue checking
+                    start = min(start, p_start)
+                    end = max(end, p_end)
+                    result.pop(i)
+                    # Don't increment i, check same position again
+                else:
+                    # No overlap, check next
+                    i += 1
+
+            if not merged:
+                result.append((start, end))

         return result
```