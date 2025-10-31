"""Test the reported bug"""
import argparse

# Simulate the pooch module not being installed
pooch = None

def download_all(path=None):
    """Simulating the actual download_all function"""
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")

def main():
    """Simulating the actual main function that has the bug"""
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    parser.add_argument("path", nargs='?', type=str,
                        default=pooch.os_cache('scipy-data'),  # This line causes the bug
                        help="Directory path to download all the data files.")
    args = parser.parse_args()
    download_all(args.path)

print("Testing the bug report:")
print("-" * 50)

try:
    main()
except AttributeError as e:
    print(f"Got confusing AttributeError: {e}")
except ImportError as e:
    print(f"Got helpful ImportError: {e}")

print("\nExpected behavior would be to get the helpful ImportError.")