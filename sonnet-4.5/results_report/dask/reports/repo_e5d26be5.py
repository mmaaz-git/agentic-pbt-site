"""
Minimal reproduction of the dask ResourceProfiler type mismatch bug.
This demonstrates that the _update_pids method incorrectly compares
a Process object with integer PIDs, causing the tracker process to
never be filtered out.
"""

from multiprocessing import current_process
import os

class MockProcess:
    """Mock process object similar to psutil.Process"""
    def __init__(self, pid_value):
        self.pid = pid_value

    def status(self):
        return "running"

    def children(self):
        return []

def buggy_update_pids(parent, pid):
    """
    This is the buggy implementation from dask/diagnostics/profile.py
    Line 252: p.pid != pid where pid is a Process object, not an integer
    """
    return [parent] + [
        p for p in parent.children() if p.pid != pid and p.status() != "zombie"
    ]

def fixed_update_pids(parent, pid):
    """
    This is the corrected implementation that compares integers
    """
    return [parent] + [
        p for p in parent.children() if p.pid != pid.pid and p.status() != "zombie"
    ]

def main():
    print("=== Demonstrating dask ResourceProfiler Type Mismatch Bug ===\n")

    # Get the current process (this is what happens on line 261)
    tracker_process = current_process()
    print(f"Type of current_process(): {type(tracker_process)}")
    print(f"PID value: {tracker_process.pid}")
    print(f"Process object: {tracker_process}\n")

    # Create mock parent process with some children
    parent = MockProcess(os.getppid())

    # Create children including one with the tracker's PID
    child1 = MockProcess(12345)
    child2 = MockProcess(tracker_process.pid)  # Same PID as tracker
    child3 = MockProcess(67890)

    # Mock the parent.children() method
    parent.children = lambda: [child1, child2, child3]

    print("Parent process children PIDs:")
    for child in parent.children():
        print(f"  - PID {child.pid}")
    print()

    # Test buggy implementation (passing Process object)
    print("=== Buggy Implementation ===")
    print("Calling _update_pids with Process object (line 271: self._update_pids(pid))")
    buggy_result = buggy_update_pids(parent, tracker_process)
    buggy_pids = [p.pid for p in buggy_result]
    print(f"Resulting PIDs: {buggy_pids}")

    # Check if tracker PID was filtered out
    if tracker_process.pid in buggy_pids:
        print(f"❌ BUG: Tracker process (PID {tracker_process.pid}) was NOT filtered out!")
        print("   This means the profiler is measuring its own resource usage.\n")
    else:
        print(f"✓ Tracker process (PID {tracker_process.pid}) was filtered out\n")

    # Test fixed implementation (passing Process.pid)
    print("=== Fixed Implementation ===")
    print("Calling _update_pids with Process.pid integer")
    fixed_result = fixed_update_pids(parent, tracker_process)
    fixed_pids = [p.pid for p in fixed_result]
    print(f"Resulting PIDs: {fixed_pids}")

    if tracker_process.pid in fixed_pids:
        print(f"✗ Tracker process (PID {tracker_process.pid}) was NOT filtered out\n")
    else:
        print(f"✓ CORRECT: Tracker process (PID {tracker_process.pid}) was filtered out\n")

    # Demonstrate the type comparison issue
    print("=== Type Comparison Analysis ===")
    test_pid = tracker_process.pid
    print(f"Integer PID value: {test_pid}")
    print(f"test_pid != tracker_process (int != Process): {test_pid != tracker_process}")
    print(f"test_pid != tracker_process.pid (int != int): {test_pid != tracker_process.pid}")
    print("\nThe bug occurs because p.pid (integer) is compared to pid (Process object),")
    print("which will ALWAYS return True, even when the PIDs match.")

if __name__ == "__main__":
    main()