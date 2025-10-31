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
    default_simplex = simplex_indices_default[i]
    bruteforce_simplex = simplex_indices_bruteforce[i]
    # Also check individual query
    individual_simplex = tri.find_simplex(points[i:i+1])[0]
    print(f"Point {i}: is_vertex={is_vertex}, default_simplex={default_simplex}, bruteforce_simplex={bruteforce_simplex}, individual_query={individual_simplex}")

print(f"\nCoplanar points: {tri.coplanar}")
print(f"Number of simplices: {len(tri.simplices)}")
print(f"All input points appear as vertices: {all(i in tri.simplices.flatten() for i in range(8))}")