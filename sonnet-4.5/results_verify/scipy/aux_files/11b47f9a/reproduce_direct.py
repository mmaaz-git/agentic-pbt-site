import numpy as np
import scipy.stats as stats

arr = np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
score = 0.0

percentile = stats.percentileofscore(arr, score)

print(f"Percentile: {percentile}")
print(f"Percentile > 100: {percentile > 100}")
print(f"Exact value: {percentile!r}")

# Test all kind parameters
for kind in ['rank', 'weak', 'strict', 'mean']:
    pct = stats.percentileofscore(arr, score, kind=kind)
    print(f"kind='{kind}': {pct!r}, > 100: {pct > 100}")