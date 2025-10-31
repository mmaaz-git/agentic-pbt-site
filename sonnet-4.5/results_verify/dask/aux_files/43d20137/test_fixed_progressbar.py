import sys
from io import StringIO

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.diagnostics.progress import ProgressBar
from dask.callbacks import Callback
from timeit import default_timer
import contextlib

# Create a fixed version of ProgressBar
class FixedProgressBar(Callback):
    """Fixed version of ProgressBar that handles missing state keys gracefully"""

    def __init__(self, minimum=0, width=40, dt=0.1, out=None):
        if out is None:
            out = sys.stdout
        self._minimum = minimum
        self._width = width
        self._dt = dt
        self._file = out
        self.last_duration = 0

    def _update_bar(self, elapsed):
        s = self._state
        if not s:
            self._draw_bar(0, elapsed)
            return
        # FIX: Use .get() with default empty list instead of direct key access
        ndone = len(s.get("finished", []))
        ntasks = sum(
            len(s.get(k, []))
            for k in ["ready", "waiting", "running"]
        ) + ndone
        if ndone < ntasks:
            self._draw_bar(ndone / ntasks if ntasks else 0, elapsed)

    def _draw_bar(self, frac, elapsed):
        from dask.utils import format_time

        bar = "#" * int(self._width * frac)
        percent = int(100 * frac)
        elapsed = format_time(elapsed)
        msg = "\r[{0:<{1}}] | {2}% Completed | {3}".format(
            bar, self._width, percent, elapsed
        )
        with contextlib.suppress(ValueError):
            if self._file is not None:
                self._file.write(msg)
                self._file.flush()

def test_fixed_version():
    """Test that the fixed version handles incomplete state dicts properly"""

    test_cases = [
        ({"finished": ["task1", "task2"], "waiting": [], "running": []}, "missing 'ready'"),
        ({"ready": ["task1"], "waiting": []}, "missing 'finished' and 'running'"),
        ({"finished": []}, "missing 'ready', 'waiting', and 'running'"),
        ({}, "empty dict"),
        ({"finished": [], "ready": [], "waiting": [], "running": []}, "all keys present"),
    ]

    print("Testing fixed ProgressBar implementation:")
    print("=" * 50)

    all_passed = True

    for state, description in test_cases:
        pbar = FixedProgressBar(out=StringIO())
        pbar._start_time = 0
        pbar._state = state

        try:
            pbar._update_bar(elapsed=1.0)
            print(f"✓ Passed with {description}")
        except Exception as e:
            print(f"✗ Failed with {description}: {e}")
            all_passed = False

    # Test None state
    pbar = FixedProgressBar(out=StringIO())
    pbar._start_time = 0
    pbar._state = None
    try:
        pbar._update_bar(elapsed=1.0)
        print(f"✓ Passed with None state")
    except Exception as e:
        print(f"✗ Failed with None state: {e}")
        all_passed = False

    print("=" * 50)
    if all_passed:
        print("✓ All tests passed with the fixed implementation!")
    else:
        print("✗ Some tests failed with the fixed implementation")

    return all_passed

if __name__ == "__main__":
    test_fixed_version()