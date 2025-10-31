"""
Hypothesis test to verify scipy.datasets._download_all.main()
provides helpful error messages when pooch is not installed.
"""
from hypothesis import given, strategies as st, example
import argparse
import sys
import io
from contextlib import redirect_stdout


class MockPooch:
    """Mock pooch module with os_cache method"""
    @staticmethod
    def os_cache(name):
        return f"/mock/cache/{name}"


@given(st.booleans())
@example(False)  # Explicitly test the failing case
def test_main_gives_helpful_error_without_pooch(has_pooch):
    """Test that main() gives a helpful ImportError when pooch is missing"""
    # Set up pooch as either a mock or None
    pooch = MockPooch() if has_pooch else None

    def download_all(path=None):
        """Same logic as scipy.datasets._download_all.download_all"""
        if pooch is None:
            raise ImportError("Missing optional dependency 'pooch' required "
                              "for scipy.datasets module. Please use pip or "
                              "conda to install 'pooch'.")
        if path is None:
            path = pooch.os_cache('scipy-data')
        return f"Would download to: {path}"

    def main():
        """Same logic as scipy.datasets._download_all.main"""
        parser = argparse.ArgumentParser(description='Download SciPy data files.')
        parser.add_argument("path", nargs='?', type=str,
                            default=pooch.os_cache('scipy-data'),  # BUG: This line causes AttributeError when pooch is None
                            help="Directory path to download all the data files.")
        args = parser.parse_args([])  # Empty args for testing
        return download_all(args.path)

    if has_pooch:
        # With pooch, it should work fine
        try:
            # Capture output to avoid test noise
            result = main()
            # Test passes - main() executed without error
        except Exception as e:
            assert False, f"Should not have raised an error with pooch installed, got: {e}"
    else:
        # Without pooch, we expect a helpful ImportError
        try:
            main()
            assert False, "Should have raised an error without pooch"
        except ImportError as e:
            # Good! Got the helpful error
            assert "pooch" in str(e).lower(), f"ImportError should mention 'pooch', got: {e}"
            assert "pip" in str(e).lower() or "conda" in str(e).lower(), f"ImportError should mention installation method, got: {e}"
        except AttributeError as e:
            # Bad! Got confusing AttributeError instead
            assert False, f"Got confusing AttributeError instead of helpful ImportError: {e}"
        except Exception as e:
            assert False, f"Got unexpected error type {type(e).__name__}: {e}"


# Run the hypothesis test
if __name__ == "__main__":
    print("Running Hypothesis test for scipy.datasets._download_all.main() error handling...")
    print("=" * 70)
    test_main_gives_helpful_error_without_pooch()