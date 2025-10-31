from dask.diagnostics import CacheProfiler


def custom_metric(value):
    return len(str(value))


# Test with empty string metric_name
prof = CacheProfiler(metric=custom_metric, metric_name="")

print(f"Expected: metric_name = ''")
print(f"Actual:   metric_name = '{prof._metric_name}'")

# This assertion should pass if the bug is fixed
assert prof._metric_name == "", f"Expected empty string, got '{prof._metric_name}'"