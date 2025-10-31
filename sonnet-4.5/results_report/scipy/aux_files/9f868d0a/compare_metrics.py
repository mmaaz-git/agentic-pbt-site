from scipy.spatial.distance import dice, jaccard, hamming, rogerstanimoto, sokalsneath
import numpy as np
import warnings

u = np.array([False, False, False])
v = np.array([False, False, False])

print("Testing dissimilarity metrics with all-False identical vectors:")
print("Input: u = [False, False, False], v = [False, False, False]")
print("-" * 60)

metrics = [
    ("dice", dice),
    ("jaccard", jaccard),
    ("hamming", hamming),
    ("rogerstanimoto", rogerstanimoto),
    ("sokalsneath", sokalsneath)
]

for name, func in metrics:
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(u, v)
            if w:
                print(f"{name:15}: {result} (Warning: {w[0].message})")
            else:
                print(f"{name:15}: {result}")
    except Exception as e:
        print(f"{name:15}: Error - {e}")