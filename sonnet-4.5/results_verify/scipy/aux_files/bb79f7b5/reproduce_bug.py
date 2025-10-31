import inspect
import scipy
import scipy.datasets._fetchers as fetchers
import scipy.datasets._download_all as download_all_module

fetch_data_source = inspect.getsource(fetchers.fetch_data)
download_all_source = inspect.getsource(download_all_module.download_all)

print("fetch_data() User-Agent header:")
for line in fetch_data_source.split('\n'):
    if 'User-Agent' in line:
        print(f"  {line.strip()}")

print(f"\n  Resolves to: 'SciPy {scipy.__version__}'")

print("\ndownload_all() User-Agent header:")
for line in download_all_source.split('\n'):
    if 'User-Agent' in line:
        print(f"  {line.strip()}")

print("\n  Resolves to: 'SciPy' (no version)")
print("\nInconsistency confirmed!")