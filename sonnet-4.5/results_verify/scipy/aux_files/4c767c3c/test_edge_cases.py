from scipy.odr._odrpack import _report_error

# Test various info values that would result in I[0] >= 7
test_values = [70000, 80000, 90000, 75432, 89999]

for info in test_values:
    I0 = info // 10000 % 10
    result = _report_error(info)
    print(f"info={info}, I[0]={I0}, result={result}, len={len(result)}")

# Also test some that work
working_values = [60000, 50000, 40000, 30000, 20000, 10000, 0]
print("\nWorking values:")
for info in working_values:
    I0 = info // 10000 % 10
    result = _report_error(info)
    print(f"info={info}, I[0]={I0}, result={result}, len={len(result)}")
