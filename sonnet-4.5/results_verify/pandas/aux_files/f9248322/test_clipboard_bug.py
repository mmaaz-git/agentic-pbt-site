#!/usr/bin/env python3
"""Test script to reproduce the pandas clipboard pbcopy/pbpaste bug"""

import subprocess
import sys

# First, let's verify what happens when we run pbcopy with "w" argument
print("Testing pbcopy with 'w' argument...")
try:
    result = subprocess.run(["pbcopy", "w"],
                          input=b"test",
                          capture_output=True,
                          timeout=1)
    print(f"pbcopy with 'w': returncode={result.returncode}")
    if result.stderr:
        print(f"stderr: {result.stderr.decode()}")
except Exception as e:
    print(f"Exception running pbcopy with 'w': {e}")

print("\n" + "="*50 + "\n")

# Test pbcopy without arguments (correct usage)
print("Testing pbcopy without arguments (correct usage)...")
try:
    result = subprocess.run(["pbcopy"],
                          input=b"test",
                          capture_output=True,
                          timeout=1)
    print(f"pbcopy without args: returncode={result.returncode}")
    if result.stderr:
        print(f"stderr: {result.stderr.decode()}")
except Exception as e:
    print(f"Exception running pbcopy: {e}")

print("\n" + "="*50 + "\n")

# Test pbpaste with "r" argument
print("Testing pbpaste with 'r' argument...")
try:
    result = subprocess.run(["pbpaste", "r"],
                          capture_output=True,
                          timeout=1)
    print(f"pbpaste with 'r': returncode={result.returncode}")
    if result.stderr:
        print(f"stderr: {result.stderr.decode()}")
except Exception as e:
    print(f"Exception running pbpaste with 'r': {e}")

print("\n" + "="*50 + "\n")

# Test pbpaste without arguments (correct usage)
print("Testing pbpaste without arguments (correct usage)...")
try:
    result = subprocess.run(["pbpaste"],
                          capture_output=True,
                          timeout=1)
    print(f"pbpaste without args: returncode={result.returncode}")
    print(f"stdout: {result.stdout.decode()[:100]}...")  # First 100 chars
except Exception as e:
    print(f"Exception running pbpaste: {e}")

print("\n" + "="*50 + "\n")

# Now test the actual pandas clipboard functions
print("Testing pandas clipboard functions...")
try:
    import pandas.io.clipboard as clipboard

    # Test init_osx_pbcopy_clipboard directly
    copy_func, paste_func = clipboard.init_osx_pbcopy_clipboard()

    print("Attempting to copy 'Hello, World!' using pandas clipboard...")
    try:
        copy_func("Hello, World!")
        print("Copy succeeded!")
    except Exception as e:
        print(f"Copy failed with error: {e}")

    print("\nAttempting to paste using pandas clipboard...")
    try:
        result = paste_func()
        print(f"Paste result: {result}")
    except Exception as e:
        print(f"Paste failed with error: {e}")

except ImportError as e:
    print(f"Could not import pandas.io.clipboard: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\n" + "="*50 + "\n")

# Test with hypothesis if available
print("Testing with Hypothesis...")
try:
    from hypothesis import given, strategies as st, settings
    import pandas.io.clipboard as clipboard

    @given(st.text())
    @settings(max_examples=10)
    def test_osx_pbcopy_round_trip(text):
        copy_func, paste_func = clipboard.init_osx_pbcopy_clipboard()
        copy_func(text)
        result = paste_func()
        assert result == text, f"Expected '{text}' but got '{result}'"
        return True

    # Run a simple test
    test_osx_pbcopy_round_trip()
    print("Hypothesis test completed successfully!")

except ImportError:
    print("Hypothesis not installed, skipping property-based test")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

print("\nTest complete!")