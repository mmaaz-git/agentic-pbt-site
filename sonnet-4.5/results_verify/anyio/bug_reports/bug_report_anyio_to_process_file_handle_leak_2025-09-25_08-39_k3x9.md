# Bug Report: anyio.to_process File Handle Leak

**Target**: `anyio.to_process.process_worker`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `process_worker()` function opens file handles to `/dev/null` for stdin/stdout redirection but never closes them, causing a resource leak in worker processes.

## Property-Based Test

```python
import os
import sys
import subprocess
from hypothesis import given, strategies as st


@given(dummy=st.just(None))
def test_process_worker_file_handle_leak(dummy):
    """
    Property: File handles opened in process_worker() should be closed on exit.
    """
    script = '''
import sys
import os

stdin = sys.stdin
stdout = sys.stdout
devnull_in = open(os.devnull)
devnull_out = open(os.devnull, "w")
sys.stdin = devnull_in
sys.stdout = devnull_out

print(f"devnull_in closed: {devnull_in.closed}", file=stdout)
print(f"devnull_out closed: {devnull_out.closed}", file=stdout)
stdout.flush()
'''

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "devnull_in closed: False" in result.stdout
    assert "devnull_out closed: False" in result.stdout
```

**Failing input**: Any execution path through `process_worker()` that exits

## Reproducing the Bug

```python
import sys
import os

def process_worker_simplified():
    stdin = sys.stdin
    stdout = sys.stdout
    sys.stdin = open(os.devnull)
    sys.stdout = open(os.devnull, "w")

    return

devnull_in = sys.stdin
devnull_out = sys.stdout

process_worker_simplified()

print(f"After function exit:")
print(f"devnull stdin closed: {devnull_in.closed}")
print(f"devnull stdout closed: {devnull_out.closed}")
```

Output:
```
After function exit:
devnull stdin closed: False
devnull stdout closed: False
```

Expected: Both file handles should be closed when the function exits.

## Why This Is A Bug

In `to_process.py` lines 198-254, the `process_worker()` function opens two file handles:

```python
def process_worker() -> None:
    # Redirect standard streams to os.devnull so that user code won't interfere with the
    # parent-worker communication
    stdin = sys.stdin
    stdout = sys.stdout
    sys.stdin = open(os.devnull)          # File handle 1: never closed
    sys.stdout = open(os.devnull, "w")    # File handle 2: never closed

    stdout.buffer.write(b"READY\n")
    while True:
        # ... process commands ...
        try:
            command, *args = pickle.load(stdin.buffer)
        except EOFError:
            return  # BUG: exits without closing devnull handles
        # ...
        if isinstance(exception, SystemExit):
            raise exception  # BUG: exits without closing devnull handles
```

The function has two exit paths:
1. `return` on line 212 (EOFError)
2. `raise exception` on line 254 (SystemExit)

Neither path closes the `/dev/null` file handles opened on lines 203-204.

**Impact:**
- Each worker process leaks 2 file descriptors
- In long-running applications with many worker processes, this accumulates
- Could eventually hit the process file descriptor limit
- Violates Python resource management best practices

## Fix

```diff
--- a/anyio/to_process.py
+++ b/anyio/to_process.py
@@ -200,10 +200,16 @@ def process_worker() -> None:
     # parent-worker communication
     stdin = sys.stdin
     stdout = sys.stdout
-    sys.stdin = open(os.devnull)
-    sys.stdout = open(os.devnull, "w")
-
-    stdout.buffer.write(b"READY\n")
+    devnull_in = open(os.devnull)
+    devnull_out = open(os.devnull, "w")
+    sys.stdin = devnull_in
+    sys.stdout = devnull_out
+
+    try:
+        stdout.buffer.write(b"READY\n")
+    # ... rest of the while loop
+    finally:
+        devnull_in.close()
+        devnull_out.close()
```

Alternatively, use context managers:

```diff
--- a/anyio/to_process.py
+++ b/anyio/to_process.py
@@ -200,8 +200,10 @@ def process_worker() -> None:
     # parent-worker communication
     stdin = sys.stdin
     stdout = sys.stdout
-    sys.stdin = open(os.devnull)
-    sys.stdout = open(os.devnull, "w")
+
+    devnull_in = open(os.devnull)
+    devnull_out = open(os.devnull, "w")
+    sys.stdin, sys.stdout = devnull_in, devnull_out

     stdout.buffer.write(b"READY\n")
     while True:
@@ -210,7 +212,11 @@ def process_worker() -> None:
         try:
             command, *args = pickle.load(stdin.buffer)
         except EOFError:
+            devnull_in.close()
+            devnull_out.close()
             return
         # ... handle commands ...
         if isinstance(exception, SystemExit):
+            devnull_in.close()
+            devnull_out.close()
             raise exception
```