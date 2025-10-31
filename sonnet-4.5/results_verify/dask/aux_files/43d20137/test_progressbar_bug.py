import sys
from io import StringIO

from hypothesis import given, strategies as st

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.diagnostics.progress import ProgressBar

# First, let's reproduce the exact bug from the report
def test_reproduce_exact_bug():
    """Test the exact reproduction case from the bug report"""
    pbar = ProgressBar(out=StringIO())
    pbar._start_time = 0
    pbar._state = {
        "finished": ["task1", "task2"],
        "waiting": [],
        "running": []
    }

    try:
        pbar._update_bar(elapsed=1.0)
        print("ERROR: Should have raised KeyError but didn't")
        return False
    except KeyError as e:
        print(f"Confirmed: KeyError raised as expected: {e}")
        return True

# Property-based test from the bug report
@given(st.sets(
    st.sampled_from(["finished", "ready", "waiting", "running"]),
    min_size=1,
    max_size=3
))
def test_progressbar_update_crashes_with_incomplete_state(present_keys):
    pbar = ProgressBar(out=StringIO())
    pbar._start_time = 0
    pbar._state = {key: [] for key in present_keys}

    required_keys = {"finished", "ready", "waiting", "running"}
    missing_keys = required_keys - present_keys

    if missing_keys:
        try:
            pbar._update_bar(elapsed=1.0)
            assert False, f"Expected KeyError for missing keys: {missing_keys}"
        except KeyError as e:
            assert str(e).strip("'") in missing_keys
    else:
        pbar._update_bar(elapsed=1.0)

# Test with complete state dict (should work)
def test_with_complete_state():
    """Test that it works with a complete state dict"""
    pbar = ProgressBar(out=StringIO())
    pbar._start_time = 0
    pbar._state = {
        "finished": ["task1", "task2"],
        "ready": ["task3"],
        "waiting": ["task4"],
        "running": ["task5"]
    }

    try:
        pbar._update_bar(elapsed=1.0)
        print("SUCCESS: No error with complete state dict")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception with complete state: {e}")
        return False

# Test with empty state dict (should be handled by existing defensive check)
def test_with_empty_state():
    """Test that empty state dict is handled properly"""
    pbar = ProgressBar(out=StringIO())
    pbar._start_time = 0
    pbar._state = {}

    try:
        pbar._update_bar(elapsed=1.0)
        print("SUCCESS: Empty state dict handled properly")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception with empty state: {e}")
        return False

# Test with None state (should be handled by existing defensive check)
def test_with_none_state():
    """Test that None state is handled properly"""
    pbar = ProgressBar(out=StringIO())
    pbar._start_time = 0
    pbar._state = None

    try:
        pbar._update_bar(elapsed=1.0)
        print("SUCCESS: None state handled properly")
        return True
    except Exception as e:
        print(f"ERROR: Unexpected exception with None state: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing ProgressBar._update_bar with incomplete state dicts")
    print("=" * 60)

    print("\n1. Testing exact reproduction case (missing 'ready' key):")
    test_reproduce_exact_bug()

    print("\n2. Testing with complete state dict:")
    test_with_complete_state()

    print("\n3. Testing with empty state dict:")
    test_with_empty_state()

    print("\n4. Testing with None state:")
    test_with_none_state()

    print("\n5. Running property-based tests:")
    # Run a few examples from the property test
    test_cases = [
        {"finished", "waiting", "running"},  # missing "ready"
        {"ready", "waiting"},  # missing "finished" and "running"
        {"finished"},  # missing three keys
    ]

    for present_keys in test_cases:
        print(f"\n   Testing with keys: {present_keys}")
        pbar = ProgressBar(out=StringIO())
        pbar._start_time = 0
        pbar._state = {key: [] for key in present_keys}

        required_keys = {"finished", "ready", "waiting", "running"}
        missing_keys = required_keys - present_keys

        if missing_keys:
            try:
                pbar._update_bar(elapsed=1.0)
                print(f"   ✗ Expected KeyError for missing keys: {missing_keys}")
            except KeyError as e:
                print(f"   ✓ KeyError raised as expected for missing '{e}'")
        else:
            try:
                pbar._update_bar(elapsed=1.0)
                print(f"   ✓ No error with all keys present")
            except Exception as e:
                print(f"   ✗ Unexpected error: {e}")