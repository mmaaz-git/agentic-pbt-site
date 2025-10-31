"""
Minimal reproduction of the scipy.datasets._download_all bug
when pooch is not installed.
"""
import argparse

# Simulate pooch not being installed
pooch = None


def download_all(path=None):
    """Same logic as scipy.datasets._download_all.download_all"""
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    if path is None:
        path = pooch.os_cache('scipy-data')
    print(f"Would download to: {path}")


def main():
    """Same logic as scipy.datasets._download_all.main"""
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    parser.add_argument("path", nargs='?', type=str,
                        default=pooch.os_cache('scipy-data'),  # BUG: This line causes AttributeError
                        help="Directory path to download all the data files.")
    args = parser.parse_args()
    download_all(args.path)


if __name__ == "__main__":
    try:
        main()
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except ImportError as e:
        print(f"ImportError: {e}")