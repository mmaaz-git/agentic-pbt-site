"""Test script to reproduce the pandas.io.sas format detection bug"""

from hypothesis import given, settings, strategies as st
import string
import pytest
from pandas.io.sas import read_sas

# First test the hypothesis-based property test
@given(
    base_name=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=20),
    middle_ext=st.sampled_from(['.xpt', '.sas7bdat']),
    suffix=st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=10)
)
@settings(max_examples=200)
def test_format_detection_false_positives(base_name, middle_ext, suffix):
    filename = base_name + middle_ext + "." + suffix
    print(f"Testing filename: {filename}")

    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        read_sas(filename)

# Direct test of the bug
def test_direct_reproduction():
    print("\n=== Direct Bug Reproduction ===")

    test_cases = [
        "data.xpt.backup",
        "file.sas7bdat.old",
        "myfile.xpt123",
        "test.xpt.bak",
        "data_v2.sas7bdat.archive"
    ]

    for filename in test_cases:
        print(f"\nTesting: {filename}")
        try:
            read_sas(filename)
            print(f"  ERROR: No exception raised for {filename}")
        except FileNotFoundError as e:
            print(f"  BUG CONFIRMED: File '{filename}' was incorrectly detected as a valid SAS format")
            print(f"  FileNotFoundError: {e}")
        except ValueError as e:
            if "unable to infer format" in str(e):
                print(f"  CORRECT: Format detection failed appropriately with ValueError")
                print(f"  Message: {e}")
            else:
                print(f"  UNEXPECTED ValueError: {e}")
        except Exception as e:
            print(f"  Unexpected error type {type(e).__name__}: {e}")

if __name__ == "__main__":
    # Run direct tests
    test_direct_reproduction()

    # Run a few hypothesis examples manually
    print("\n=== Manual Hypothesis Examples ===")
    test_filenames = [
        "test.xpt.backup",
        "abc.sas7bdat.123",
        "file.xpt.old"
    ]

    for filename in test_filenames:
        print(f"\nTesting: {filename}")
        try:
            read_sas(filename)
            print(f"  ERROR: No exception raised")
        except FileNotFoundError:
            print(f"  BUG CONFIRMED: Incorrectly detected as valid format")
        except ValueError as e:
            if "unable to infer format" in str(e):
                print(f"  CORRECT: Format detection failed appropriately")
            else:
                print(f"  Unexpected ValueError: {e}")