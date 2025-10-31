from hypothesis import given, strategies as st
import anyio
from anyio._core._synchronization import Condition


@given(st.integers(min_value=1, max_value=10))
def test_notify_exactly_n_property(n):
    condition = Condition()

    num_waiters = n - 1
    for _ in range(num_waiters):
        condition._waiters.append(anyio.Event())

    notified_count = 0
    for event in list(condition._waiters):
        if event.is_set():
            notified_count += 1

    condition.notify(n)

    actual_notified = sum(1 for e in list(condition._waiters)[:num_waiters] if e.is_set())
    assert actual_notified == n, f"Expected exactly {n} notifications, got {actual_notified}"

if __name__ == "__main__":
    # Test with specific case mentioned
    test_notify_exactly_n_property(5)
    print("Test passed!")