#!/usr/bin/env python3
"""Test to reproduce the reported bug in dask.diagnostics.profile._Tracker"""

from multiprocessing import current_process

# First, let's verify the basic type mismatch issue
proc = current_process()
child_pid = 12345

print("=== Basic Type Mismatch Test ===")
buggy_comparison = child_pid != proc
correct_comparison = child_pid != proc.pid

print(f"Process object: {proc}")
print(f"Process PID: {proc.pid}")
print(f"Child PID to compare: {child_pid}")
print(f"Buggy comparison (int != Process): {child_pid} != {proc} = {buggy_comparison}")
print(f"Correct comparison (int != int): {child_pid} != {proc.pid} = {correct_comparison}")
print(f"Type of proc: {type(proc)}")
print(f"Type of proc.pid: {type(proc.pid)}")

# Now let's test the actual bug in context
print("\n=== Testing actual _Tracker implementation ===")
try:
    from dask.diagnostics.profile import _Tracker
    from unittest.mock import Mock
    from hypothesis import given, strategies as st

    # Create a simple test without hypothesis first
    tracker = _Tracker()
    mock_parent = Mock()
    mock_child = Mock()
    mock_child.pid = 54321  # Some arbitrary PID
    mock_child.status.return_value = "running"
    mock_parent.children.return_value = [mock_child]
    tracker.parent = mock_parent

    # This mimics what happens in the run() method
    pid_object = current_process()  # This is what line 261 does
    result = tracker._update_pids(pid_object)

    print(f"Mock child PID: {mock_child.pid}")
    print(f"Current process object passed: {pid_object}")
    print(f"Result includes mock_child: {mock_child in result}")
    print(f"Result includes parent: {mock_parent in result}")

    # The bug is that p.pid (integer) != pid (Process object) is always True
    # So the child is never filtered out even if it should be

    # Let's test what happens if we pass the correct type
    pid_int = current_process().pid
    result_correct = tracker._update_pids(pid_int)
    print(f"\nWith correct integer PID ({pid_int}):")
    print(f"Result includes mock_child: {mock_child in result_correct}")

    # Now test with a child that has the same PID as current process
    mock_child_same_pid = Mock()
    mock_child_same_pid.pid = current_process().pid
    mock_child_same_pid.status.return_value = "running"
    mock_parent.children.return_value = [mock_child_same_pid]

    result_with_process_obj = tracker._update_pids(pid_object)
    result_with_int = tracker._update_pids(pid_int)

    print(f"\nWith child having same PID as current process ({pid_int}):")
    print(f"Result with Process object (buggy): {mock_child_same_pid in result_with_process_obj}")
    print(f"Result with integer (correct): {mock_child_same_pid in result_with_int}")

    # Run the hypothesis test
    print("\n=== Running Hypothesis Test ===")

    @given(st.integers(min_value=1000, max_value=65535))
    def test_update_pids_type_mismatch(tracker_pid):
        tracker = _Tracker()
        mock_parent = Mock()
        mock_child = Mock()
        mock_child.pid = tracker_pid
        mock_child.status.return_value = "running"
        mock_parent.children.return_value = [mock_child]
        tracker.parent = mock_parent

        proc_obj = current_process()
        result = tracker._update_pids(proc_obj)

        # This should always pass because of the bug
        assert mock_child in result, "Bug: child always included because int != Process is always True"

    # Run the hypothesis test
    test_update_pids_type_mismatch()
    print("Hypothesis test PASSED - confirming that children are always included due to type mismatch")

except ImportError as e:
    print(f"Could not import required modules: {e}")
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()