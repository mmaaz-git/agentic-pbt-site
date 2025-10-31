import sys
import inspect

# First verify the actual code inspection
print("=== Actual Code Inspection ===")
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

# Check if version is in the source code
print("\n=== Version Check in Source ===")
print(f"Version in fetch_data source: {('sys.modules' in fetch_source and '__version__' in fetch_source)}")
print(f"Version in download_all source: {('sys.modules' in download_source and '__version__' in download_source)}")

# Run the hypothesis test
print("\n=== Running Hypothesis Test ===")
def test_user_agent_consistency():
    import sys
    import inspect
    from scipy.datasets._fetchers import fetch_data
    from scipy.datasets._download_all import download_all

    fetch_source = inspect.getsource(fetch_data)
    download_source = inspect.getsource(download_all)

    # Check if version is included
    version_in_fetch = 'sys.modules' in fetch_source and '__version__' in fetch_source
    version_in_download = 'sys.modules' in download_source and '__version__' in download_source

    print(f"Version in fetch_data: {version_in_fetch}")
    print(f"Version in download_all: {version_in_download}")

    assert version_in_fetch, "fetch_data should include version"
    assert version_in_download, "download_all should include version"

try:
    test_user_agent_consistency()
    print("Test PASSED")
except AssertionError as e:
    print(f"Test FAILED: {e}")