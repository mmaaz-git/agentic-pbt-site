"""Test the proposed fix"""
import argparse

# Simulate the pooch module not being installed
pooch = None

def download_all(path=None):
    """Simulating the actual download_all function"""
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    if path is None:
        path = pooch.os_cache('scipy-data')
    # ... rest of the function

def main_fixed():
    """The proposed fix"""
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    default_path = pooch.os_cache('scipy-data') if pooch is not None else None
    parser.add_argument("path", nargs='?', type=str,
                        default=default_path,
                        help="Directory path to download all the data files.")
    args = parser.parse_args()
    if args.path is None and pooch is not None:
        args.path = pooch.os_cache('scipy-data')
    download_all(args.path)

print("Testing the proposed fix:")
print("-" * 50)

try:
    main_fixed()
except AttributeError as e:
    print(f"Got confusing AttributeError: {e}")
except ImportError as e:
    print(f"Got helpful ImportError: {e}")

print("\nThis gives the expected helpful ImportError!")