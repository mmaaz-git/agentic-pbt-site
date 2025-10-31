from hypothesis import given, strategies as st
from dask.diagnostics import CacheProfiler
import pytest


def custom_metric(value):
    return len(str(value))


@given(st.text())
def test_cache_profiler_custom_metric_name(metric_name):
    prof = CacheProfiler(metric=custom_metric, metric_name=metric_name)
    assert prof._metric_name == metric_name


# Direct test with empty string
def test_empty_string():
    prof = CacheProfiler(metric=custom_metric, metric_name="")
    assert prof._metric_name == ""


if __name__ == "__main__":
    # Test with empty string specifically
    try:
        test_empty_string()
        print("Test passed for empty string")
    except AssertionError as e:
        print(f"Test FAILED for empty string")
        prof = CacheProfiler(metric=custom_metric, metric_name="")
        print(f"Expected: metric_name = ''")
        print(f"Actual:   metric_name = '{prof._metric_name}'")

    # Run hypothesis tests
    import hypothesis.strategies as st
    test_strategy = st.text()

    # Test a few examples
    for test_value in ["", "test", None, "   "]:
        if test_value is not None:
            try:
                prof = CacheProfiler(metric=custom_metric, metric_name=test_value)
                print(f"metric_name='{test_value}' -> _metric_name='{prof._metric_name}'")
                assert prof._metric_name == test_value
                print(f"  ✓ Test passed")
            except AssertionError:
                print(f"  ✗ Test FAILED - expected '{test_value}' but got '{prof._metric_name}'")