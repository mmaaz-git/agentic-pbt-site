from hypothesis import given, strategies as st
from dask.diagnostics import CacheProfiler


def custom_metric(value):
    return len(str(value))


@given(st.text())
def test_cache_profiler_custom_metric_name(metric_name):
    prof = CacheProfiler(metric=custom_metric, metric_name=metric_name)
    assert prof._metric_name == metric_name


# Run the test
if __name__ == "__main__":
    test_cache_profiler_custom_metric_name()