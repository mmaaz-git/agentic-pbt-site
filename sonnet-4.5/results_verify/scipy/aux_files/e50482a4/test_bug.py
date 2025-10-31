import scipy.stats as stats
import numpy as np

# Test the reported case
data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
score = 1.0

result = stats.percentileofscore(data, score, kind='rank')
print(f"Result: {result}")
print(f"Result > 100: {result > 100}")
print(f"Result == 100: {result == 100}")
print(f"Result - 100: {result - 100}")
print(f"Result repr: {repr(result)}")

# Let's also test with other kinds
print("\nTesting all kinds:")
for kind in ['rank', 'weak', 'strict', 'mean']:
    result = stats.percentileofscore(data, score, kind=kind)
    print(f"  kind={kind}: {result}, exceeds 100? {result > 100}")