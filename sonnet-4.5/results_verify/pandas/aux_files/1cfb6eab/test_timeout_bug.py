import pandas.io.clipboard as clipboard
import time
from hypothesis import given, strategies as st, settings

# Test 1: Direct reproduction test
def test_direct_reproduction():
    """Test the bug directly with the actual waitForPaste function"""

    # Mock the paste function to always return empty string
    original_paste = clipboard.paste
    clipboard.paste = lambda: ""

    try:
        # Test with timeout=0
        start = time.time()
        try:
            clipboard.waitForPaste(timeout=0)
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start
            print(f"timeout=0: Timed out after {elapsed:.4f}s")
            assert elapsed > 0.005, f"Expected wait of >0.005s, but got {elapsed:.4f}s"

        # Test with timeout=-1
        start = time.time()
        try:
            clipboard.waitForPaste(timeout=-1)
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start
            print(f"timeout=-1: Timed out after {elapsed:.4f}s")
            assert elapsed > 0.005, f"Expected wait of >0.005s, but got {elapsed:.4f}s"

        # Test with timeout=0.001
        start = time.time()
        try:
            clipboard.waitForPaste(timeout=0.001)
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start
            print(f"timeout=0.001: Timed out after {elapsed:.4f}s")
            assert elapsed > 0.005, f"Expected wait of >0.005s, but got {elapsed:.4f}s"
    finally:
        clipboard.paste = original_paste


# Test 2: Test waitForNewPaste
def test_waitForNewPaste():
    """Test the bug with waitForNewPaste function"""

    # Mock the paste function to always return the same string
    original_paste = clipboard.paste
    clipboard.paste = lambda: "same_text"

    try:
        # Test with timeout=0
        start = time.time()
        try:
            clipboard.waitForNewPaste(timeout=0)
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start
            print(f"waitForNewPaste timeout=0: Timed out after {elapsed:.4f}s")
            assert elapsed > 0.005, f"Expected wait of >0.005s, but got {elapsed:.4f}s"

        # Test with timeout=-1
        start = time.time()
        try:
            clipboard.waitForNewPaste(timeout=-1)
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start
            print(f"waitForNewPaste timeout=-1: Timed out after {elapsed:.4f}s")
            assert elapsed > 0.005, f"Expected wait of >0.005s, but got {elapsed:.4f}s"
    finally:
        clipboard.paste = original_paste


# Test 3: Property-based test from bug report
@given(st.floats(min_value=-100, max_value=0.005, allow_nan=False, allow_infinity=False))
@settings(max_examples=20)  # Reduced for faster testing
def test_waitForPaste_timeout_precision(timeout):
    """Property-based test for timeout precision"""

    # Mock the paste function to always return empty string
    original_paste = clipboard.paste
    clipboard.paste = lambda: ""

    try:
        start = time.time()
        try:
            clipboard.waitForPaste(timeout)
            assert False, f"Should have raised PyperclipTimeoutException for timeout={timeout}"
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start

            if timeout <= 0:
                # Bug assertion: with timeout <= 0, we expect immediate timeout
                # But due to the bug, it will wait at least 0.01s
                print(f"timeout={timeout:.4f}: elapsed={elapsed:.4f}s (expected <0.005s)")
                assert elapsed > 0.005, f"Bug confirmed: waited {elapsed:.4f}s instead of immediate timeout"
    finally:
        clipboard.paste = original_paste


if __name__ == "__main__":
    print("=== Testing waitForPaste/waitForNewPaste timeout bug ===\n")

    print("Test 1: Direct reproduction")
    test_direct_reproduction()

    print("\nTest 2: waitForNewPaste")
    test_waitForNewPaste()

    print("\nTest 3: Property-based testing")
    test_waitForPaste_timeout_precision()

    print("\n=== All tests demonstrate the bug ===")