# Bug Report: scipy.datasets Inconsistent User-Agent Headers

**Target**: `scipy.datasets.download_all` and `scipy.datasets._fetchers.fetch_data`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`fetch_data()` and `download_all()` use inconsistent User-Agent headers when downloading datasets, despite both referencing the same GitHub issue #21879.

## Property-Based Test

```python
from unittest.mock import patch, MagicMock
import scipy.datasets._fetchers
import scipy.datasets._download_all


def test_user_agent_consistency():
    """Both fetch_data and download_all should use the same User-Agent format."""
    with patch('scipy.datasets._fetchers.pooch') as mock_pooch_fetchers, \
         patch('scipy.datasets._download_all.pooch') as mock_pooch_download:

        mock_fetcher = MagicMock()
        mock_pooch_fetchers.HTTPDownloader.return_value = MagicMock()
        mock_pooch_download.HTTPDownloader.return_value = MagicMock()
        mock_pooch_download.os_cache.return_value = '/tmp/cache'
        mock_pooch_download.retrieve = MagicMock()

        scipy.datasets._fetchers.fetch_data("test.dat", data_fetcher=mock_fetcher)
        fetch_user_agent = mock_pooch_fetchers.HTTPDownloader.call_args[1]['headers']['User-Agent']

        scipy.datasets._download_all.download_all()
        download_user_agent = mock_pooch_download.HTTPDownloader.call_args[1]['headers']['User-Agent']

        assert fetch_user_agent == download_user_agent
```

**Failing input**: Any dataset download

## Reproducing the Bug

Compare the User-Agent headers in both files:

In `_fetchers.py` (line 34):
```python
downloader = pooch.HTTPDownloader(
    headers={"User-Agent": f"SciPy {sys.modules['scipy'].__version__}"}
)
```

In `_download_all.py` (line 54):
```python
downloader = pooch.HTTPDownloader(headers={"User-Agent": "SciPy"})
```

Result:
- `fetch_data` sends: `"SciPy 1.16.2"`
- `download_all` sends: `"SciPy"`

## Why This Is A Bug

Both functions reference GitHub issue #21879, indicating the User-Agent header is significant (likely for server-side tracking or rate limiting). Inconsistent headers between functions that perform the same operation (downloading scipy datasets) can cause:

1. Inconsistent server-side behavior
2. Difficulty tracking download statistics
3. Potential debugging confusion

## Fix

Make `download_all` use the same User-Agent format as `fetch_data`:

```diff
--- a/scipy/datasets/_download_all.py
+++ b/scipy/datasets/_download_all.py
@@ -9,6 +9,7 @@
 import argparse
 try:
     import pooch
 except ImportError:
     pooch = None
+import sys


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