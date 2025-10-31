from scipy.odr._odrpack import _report_error

result = _report_error(70000)
print(f"Result: {result}")
print(f"Length: {len(result)}")

assert len(result) > 0, "Expected non-empty list of error messages"