#!/usr/bin/env python3
"""Test that the proposed fix would work"""

from dask.diagnostics import CacheProfiler
from dask.diagnostics.profile import CacheData
from timeit import default_timer

class FixedCacheProfiler(CacheProfiler):
    """CacheProfiler with the proposed fix applied"""

    def _posttask(self, key, value, dsk, state, id):
        t = default_timer()
        self._cache[key] = (self._metric(value), t)
        for k in state["released"] & self._cache.keys():
            metric, start = self._cache.pop(k)
            # FIX: Use self._dsk[k] instead of dsk[k]
            self.results.append(CacheData(k, self._dsk[k], metric, start, t))

    def _finish(self, dsk, state, failed):
        t = default_timer()
        for k, (metric, start) in self._cache.items():
            # FIX: Use self._dsk[k] instead of dsk[k]
            self.results.append(CacheData(k, self._dsk[k], metric, start, t))
        self._cache.clear()


print("Testing original CacheProfiler (should fail):")
prof_original = CacheProfiler()

dsk1 = {"x": 1, "y": ("add", "x", 10)}
prof_original._start(dsk1)

state1 = {"released": set()}
prof_original._posttask("y", 11, dsk1, state1, 1)

dsk2 = {"z": ("mul", "y", 2)}
state2 = {"released": {"y"}}

try:
    prof_original._posttask("z", 22, dsk2, state2, 1)
    print("  No error - unexpected!")
except KeyError as e:
    print(f"  KeyError as expected: {e}")

print("\nTesting FixedCacheProfiler (should work):")
prof_fixed = FixedCacheProfiler()

dsk1 = {"x": 1, "y": ("add", "x", 10)}
prof_fixed._start(dsk1)

state1 = {"released": set()}
prof_fixed._posttask("y", 11, dsk1, state1, 1)

dsk2 = {"z": ("mul", "y", 2)}
state2 = {"released": {"y"}}

try:
    prof_fixed._posttask("z", 22, dsk2, state2, 1)
    print("  Success - no error!")
    print(f"  Results: {prof_fixed.results}")
    print(f"  Result shows task 'y' = {prof_fixed.results[0].task}")
except Exception as e:
    print(f"  Unexpected error: {e}")

print("\nTesting _finish method:")
prof_fixed2 = FixedCacheProfiler()

dsk1 = {"a": 1, "b": ("add", "a", 10)}
prof_fixed2._start(dsk1)
prof_fixed2._posttask("a", 1, dsk1, {"released": set()}, 1)
prof_fixed2._posttask("b", 11, dsk1, {"released": set()}, 1)

# Now call _finish with a different dsk
dsk2 = {"c": 100}
try:
    prof_fixed2._finish(dsk2, {}, False)
    print("  _finish succeeded with fixed version")
    print(f"  Results contain {len(prof_fixed2.results)} entries")
except Exception as e:
    print(f"  Error in _finish: {e}")