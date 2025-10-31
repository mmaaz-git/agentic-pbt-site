import numpy as np
from scipy.spatial import Delaunay

np.random.seed(0)
points = np.random.randn(8, 4) * 100

tri = Delaunay(points)

simplex_indices_default = tri.find_simplex(points)
simplex_indices_bruteforce = tri.find_simplex(points, bruteforce=True)

print("Default algorithm:", simplex_indices_default)
print("Bruteforce algorithm:", simplex_indices_bruteforce)
print(f"\nPoints 6 and 7 return -1 with default but are found with bruteforce")

for i in [6, 7]:
    is_vertex = np.any(tri.simplices == i)
    print(f"Point {i}: is_vertex={is_vertex}, default_simplex={simplex_indices_default[i]}, bruteforce_simplex={simplex_indices_bruteforce[i]}")

# Test the suggested workaround
print("\nTesting with increased tolerance tol=1e-10:")
simplex_indices_tol = tri.find_simplex(points, tol=1e-10)
print("With tol=1e-10:", simplex_indices_tol)
print(f"All points found with tol=1e-10: {np.all(simplex_indices_tol >= 0)}")