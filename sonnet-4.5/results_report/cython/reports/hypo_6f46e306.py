#!/usr/bin/env python3
"""Property-based test for Cython.TestUtils.write_newer_file using Hypothesis."""

import sys
import os
import tempfile
import signal
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck

# Add Cython to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.TestUtils import write_newer_file

# Set a timeout to prevent infinite loop from hanging forever
def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out after 5 seconds - infinite loop detected!")

@given(st.text(min_size=0, max_size=100))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None, max_examples=5)
def test_write_newer_file_terminates(content):
    """Test that write_newer_file terminates when newer_than file doesn't exist."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout per test

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            nonexistent = os.path.join(tmpdir, "nonexistent.txt")

            # This should write the file and return, not hang
            write_newer_file(file_path, nonexistent, content)

            # Verify the file was created with correct content
            assert os.path.exists(file_path)
            with open(file_path) as f:
                assert f.read() == content

            signal.alarm(0)  # Cancel alarm on success

    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        raise AssertionError(f"write_newer_file entered infinite loop with content: {repr(content)}")
    finally:
        signal.alarm(0)  # Ensure alarm is cancelled

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for write_newer_file...")
    print("Testing with various string inputs when newer_than file doesn't exist.\n")

    try:
        test_write_newer_file_terminates()
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        print("\nThis confirms the bug: write_newer_file enters an infinite loop")
        print("when the newer_than file doesn't exist.")
    else:
        print("✅ All tests passed!")
        print("This should not happen if the bug exists.")