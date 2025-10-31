import numpy as np
from scipy.spatial import Delaunay

np.random.seed(0)
points = np.random.randn(8, 4) * 100

tri = Delaunay(points)

# Test batch query
print("Batch query - find_simplex(points):")
batch_default = tri.find_simplex(points)
batch_bruteforce = tri.find_simplex(points, bruteforce=True)
print("Default:", batch_default)
print("Bruteforce:", batch_bruteforce)

# Test individual queries
print("\nIndividual queries - find_simplex(points[i:i+1]):")
individual_default = []
individual_bruteforce = []
for i in range(8):
    ind_def = tri.find_simplex(points[i:i+1])[0]
    ind_brute = tri.find_simplex(points[i:i+1], bruteforce=True)[0]
    individual_default.append(ind_def)
    individual_bruteforce.append(ind_brute)

print("Default:", individual_default)
print("Bruteforce:", individual_bruteforce)

# Test with reshape
print("\nUsing reshape - find_simplex(points[i].reshape(1, -1)):")
reshape_default = []
for i in range(8):
    res = tri.find_simplex(points[i].reshape(1, -1))[0]
    reshape_default.append(res)
print("Default:", reshape_default)

# Verify points are actually the same
print("\nVerifying points 6 and 7 are actually vertices:")
for i in [6, 7]:
    is_in_simplices = np.any(tri.simplices == i)
    print(f"Point {i}: appears in simplices = {is_in_simplices}")