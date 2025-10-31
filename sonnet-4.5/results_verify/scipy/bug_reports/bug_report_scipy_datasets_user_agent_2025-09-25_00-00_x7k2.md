# Bug Report: scipy.datasets Inconsistent User-Agent Headers

**Target**: `scipy.datasets._download_all.download_all` and `scipy.datasets._fetchers.fetch_data`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The User-Agent header sent to data servers is inconsistent: `fetch_data` includes the SciPy version while `download_all` does not, despite both functions referencing the same GitHub issue (#21879) that introduced User-Agent headers.

## Property-Based Test

```python
def test_user_agent_consistency():
    import sys
    import inspect
    from scipy.datasets._fetchers import fetch_data
    from scipy.datasets._download_all import download_all

    fetch_source = inspect.getsource(fetch_data)
    download_source = inspect.getsource(download_all)

    assert 'sys.modules[\'scipy\'].__version__' in fetch_source
    assert 'sys.modules[\'scipy\'].__version__' in download_source
```

**Failing observation**: `download_all` uses `"SciPy"` while `fetch_data` uses `f"SciPy {sys.modules['scipy'].__version__}"`

## Reproducing the Bug

```python
import sys
import inspect
from scipy.datasets._fetchers import fetch_data
from scipy.datasets._download_all import download_all

fetch_source = inspect.getsource(fetch_data)
download_source = inspect.getsource(download_all)

print("fetch_data User-Agent:")
for line in fetch_source.split('\n'):
    if 'User-Agent' in line:
        print(f"  {line.strip()}")

print("\ndownload_all User-Agent:")
for line in download_source.split('\n'):
    if 'User-Agent' in line:
        print(f"  {line.strip()}")
```

Output:
```
fetch_data User-Agent:
  headers={"User-Agent": f"SciPy {sys.modules['scipy'].__version__}"}

download_all User-Agent:
  downloader = pooch.HTTPDownloader(headers={"User-Agent": "SciPy"})
```

## Why This Is A Bug

Both functions reference the same GitHub issue (#21879) in comments, suggesting they were both updated to include User-Agent headers for the same reason. However, `fetch_data` includes the version number while `download_all` doesn't.

This inconsistency means:
1. Server logs will show different User-Agents for the same library
2. Server admins cannot reliably identify which SciPy version is making requests via `download_all`
3. The code lacks consistency, suggesting one implementation may have been updated without updating the other

Since `fetch_data` is more specific (includes version), it's likely the correct implementation, and `download_all` should be updated to match.

## Fix

```diff
--- a/scipy/datasets/_download_all.py
+++ b/scipy/datasets/_download_all.py
@@ -1,3 +1,4 @@
+import sys
 """
 Platform independent script to download all the
 `scipy.datasets` module data files.
@@ -51,7 +52,7 @@ def download_all(path=None):
     if path is None:
         path = pooch.os_cache('scipy-data')
     # https://github.com/scipy/scipy/issues/21879
-    downloader = pooch.HTTPDownloader(headers={"User-Agent": "SciPy"})
+    downloader = pooch.HTTPDownloader(headers={"User-Agent": f"SciPy {sys.modules['scipy'].__version__}"})
     for dataset_name, dataset_hash in _registry.registry.items():
         pooch.retrieve(url=_registry.registry_urls[dataset_name],
                        known_hash=dataset_hash,
```