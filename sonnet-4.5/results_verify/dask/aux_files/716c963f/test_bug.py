from multiprocessing import current_process

class MockProcess:
    def __init__(self, pid_value):
        self.pid = pid_value

    def status(self):
        return "running"

def buggy_update_pids(parent_children, pid):
    """This simulates the current dask implementation"""
    return [
        p for p in parent_children if p.pid != pid and p.status() != "zombie"
    ]

def fixed_update_pids(parent_children, pid):
    """This simulates the proposed fix"""
    return [
        p for p in parent_children if p.pid != pid.pid and p.status() != "zombie"
    ]

# Test the bug
tracker_process = current_process()
print(f"Tracker process type: {type(tracker_process)}")
print(f"Tracker process PID: {tracker_process.pid}")
print(f"Tracker PID type: {type(tracker_process.pid)}")

children = [
    MockProcess(12345),
    MockProcess(tracker_process.pid),
    MockProcess(67890),
]

print("\nBuggy version test:")
buggy_result = buggy_update_pids(children, tracker_process)
print(f"Buggy version PIDs: {[p.pid for p in buggy_result]}")

print("\nFixed version test:")
fixed_result = fixed_update_pids(children, tracker_process)
print(f"Fixed version PIDs: {[p.pid for p in fixed_result]}")

print("\nAssertions:")
print(f"Tracker PID {tracker_process.pid} in buggy result: {tracker_process.pid in [p.pid for p in buggy_result]}")
print(f"Tracker PID {tracker_process.pid} in fixed result: {tracker_process.pid in [p.pid for p in fixed_result]}")

# These are the assertions from the bug report
assert tracker_process.pid in [p.pid for p in buggy_result], "Bug: tracker PID should be in buggy result"
assert tracker_process.pid not in [p.pid for p in fixed_result], "Fixed: tracker PID should NOT be in fixed result"
print("\nAll assertions passed - bug confirmed!")