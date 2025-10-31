from hypothesis import given, strategies as st
from multiprocessing import current_process


class MockPsutilProcess:
    def __init__(self, pid):
        self.pid = pid

    def children(self):
        return [MockPsutilProcess(current_process().pid)]

    def status(self):
        return "running"


def buggy_update_pids(parent, pid):
    """This simulates the actual _update_pids method"""
    return [parent] + [
        p for p in parent.children() if p.pid != pid and p.status() != "zombie"
    ]


@given(st.integers(min_value=1, max_value=65535))
def test_tracker_filters_self(parent_pid):
    parent = MockPsutilProcess(parent_pid)

    # Buggy version - passing Process object instead of pid
    pid_wrong = current_process()
    result_buggy = buggy_update_pids(parent, pid_wrong)

    # Correct version - passing the actual pid integer
    pid_correct = current_process().pid
    result_correct = buggy_update_pids(parent, pid_correct)

    # The buggy version should include the tracker process, correct version should not
    assert len(result_buggy) != len(result_correct), \
        f"Type mismatch causes filter to fail - buggy: {len(result_buggy)}, correct: {len(result_correct)}"


# Run the test
if __name__ == "__main__":
    test_tracker_filters_self(1234)
    print("Test completed - bug confirmed!")