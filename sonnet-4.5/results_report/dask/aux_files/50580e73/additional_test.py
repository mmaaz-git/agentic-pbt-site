from dask.diagnostics import CacheProfiler


def custom_metric(value):
    return len(str(value))


# Test various metric_name inputs
test_cases = [
    ("", "empty string"),
    ("   ", "whitespace string"),
    ("test", "normal string"),
    (None, "None"),
]

for metric_name, description in test_cases:
    prof = CacheProfiler(metric=custom_metric, metric_name=metric_name)
    print(f"{description:20} | metric_name={repr(metric_name):10} -> _metric_name={repr(prof._metric_name)}")