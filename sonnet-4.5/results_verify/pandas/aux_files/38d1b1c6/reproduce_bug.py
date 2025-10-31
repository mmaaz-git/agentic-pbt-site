import platform
import traceback

import hypothesis.strategies as st
import pytest
from hypothesis import given

from pandas.errors import PyperclipWindowsException

print(f"Platform: {platform.system()}")
print(f"Python version: {platform.python_version()}")
print()

# First, run the hypothesis test
print("Running Hypothesis test...")
@given(st.text())
def test_pyperclip_windows_exception_crashes_with_any_message(message):
    if platform.system() != 'Windows':
        with pytest.raises(AttributeError):
            PyperclipWindowsException(message)
    else:
        # On Windows, it should work
        exc = PyperclipWindowsException(message)
        assert isinstance(exc, PyperclipWindowsException)

# Run the test
try:
    test_pyperclip_windows_exception_crashes_with_any_message()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Now run the simple reproduction
print("Running simple reproduction...")
try:
    exc = PyperclipWindowsException("Clipboard access denied")
    print(f"Success! Created exception: {exc}")
except AttributeError as e:
    print(f"Bug confirmed! AttributeError: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()