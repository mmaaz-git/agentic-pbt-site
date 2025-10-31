import numpy as np
from scipy.spatial import Delaunay

np.random.seed(640)
points = np.random.randn(13, 2)

tri = Delaunay(points)
point_1 = points[1]

result = tri.find_simplex(point_1)
print(f"find_simplex(point_1) = {result}")

print(f"Point 1 is vertex of simplices: ", end="")
for s_idx, simplex in enumerate(tri.simplices):
    if 1 in simplex:
        print(s_idx, end=" ")
print()

print(f"With tol=1e-12: {tri.find_simplex(point_1, tol=1e-12)}")

# Let's also test with various tolerance values
print("\nTesting with different tolerance values:")
tolerances = [None, 100*np.finfo(float).eps, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
for tol in tolerances:
    result = tri.find_simplex(point_1, tol=tol)
    print(f"  tol={tol}: result={result}")