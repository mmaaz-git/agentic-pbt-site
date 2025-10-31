from dask.diagnostics import CacheProfiler


def custom_metric(value):
    return len(str(value))


prof = CacheProfiler(metric=custom_metric, metric_name="")

print(f"Expected: metric_name = ''")
print(f"Actual:   metric_name = '{prof._metric_name}'")

assert prof._metric_name == ""