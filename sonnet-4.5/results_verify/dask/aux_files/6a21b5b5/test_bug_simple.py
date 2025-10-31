from multiprocessing import current_process


class MockPsutilProcess:
    def __init__(self, pid):
        self.pid = pid

    def children(self):
        # Return a child with the current process's PID (simulating the tracker)
        return [MockPsutilProcess(current_process().pid)]

    def status(self):
        return "running"


def buggy_update_pids(parent, pid):
    """This simulates the actual _update_pids method from line 250-253"""
    return [parent] + [
        p for p in parent.children() if p.pid != pid and p.status() != "zombie"
    ]


# Test the bug
parent = MockPsutilProcess(1234)

# Buggy version - passing Process object instead of pid (as done on line 261)
pid_wrong = current_process()  # This is a Process object
result_buggy = buggy_update_pids(parent, pid_wrong)

# Correct version - passing the actual pid integer
pid_correct = current_process().pid  # This is an integer
result_correct = buggy_update_pids(parent, pid_correct)

print(f"Current process PID: {current_process().pid}")
print(f"Type of pid_wrong: {type(pid_wrong)}")
print(f"Type of pid_correct: {type(pid_correct)}")
print()
print(f"Buggy version (Process object): {len(result_buggy)} processes")
print(f"  - Parent included: Yes")
print(f"  - Tracker process included: Yes (BUG - should be excluded)")
print()
print(f"Correct version (integer pid): {len(result_correct)} processes")
print(f"  - Parent included: Yes")
print(f"  - Tracker process included: No (correctly excluded)")
print()
print(f"Bug confirmed: {len(result_buggy) != len(result_correct)}")