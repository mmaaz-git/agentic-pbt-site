import numpy as np
import scipy.stats as stats

# Test case that causes percentileofscore to return value > 100
arr = np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])
score = 0.0

# Test with default 'rank' kind
print("Testing scipy.stats.percentileofscore with problematic input")
print(f"Array: {arr}")
print(f"Score: {score}")
print()

percentile = stats.percentileofscore(arr, score)
print(f"Result with kind='rank' (default): {percentile}")
print(f"Is percentile > 100? {percentile > 100}")
print(f"Exact value: {percentile:.20f}")
print()

# Test with all 'kind' parameters
for kind_param in ['rank', 'weak', 'strict', 'mean']:
    result = stats.percentileofscore(arr, score, kind=kind_param)
    print(f"Result with kind='{kind_param}': {result}")
    print(f"Is result > 100? {result > 100}")
    print(f"Exact value: {result:.20f}")
    print()