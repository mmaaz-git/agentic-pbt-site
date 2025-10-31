import numpy as np
from scipy.spatial import Delaunay

np.random.seed(0)
points = np.random.randn(8, 4) * 100

tri = Delaunay(points)

print("Coplanar points:", tri.coplanar)
print("Length of coplanar:", len(tri.coplanar))

# Check if points 6 and 7 are in coplanar
if len(tri.coplanar) > 0:
    coplanar_point_indices = tri.coplanar[:, 0]
    print("Indices of coplanar points:", coplanar_point_indices)
    print("Are points 6 and 7 coplanar?", 6 in coplanar_point_indices, 7 in coplanar_point_indices)

# Check if points 6 and 7 are actual vertices in the simplices
print("\nChecking if points 6 and 7 are vertices in the simplices:")
for i in [6, 7]:
    is_vertex = np.any(tri.simplices == i)
    print(f"Point {i} is a vertex: {is_vertex}")

# Let's check all points
print("\nChecking all points as vertices:")
for i in range(8):
    is_vertex = np.any(tri.simplices == i)
    default_found = tri.find_simplex(points[i:i+1])[0] >= 0
    bruteforce_found = tri.find_simplex(points[i:i+1], bruteforce=True)[0] >= 0
    print(f"Point {i}: is_vertex={is_vertex}, default_found={default_found}, bruteforce_found={bruteforce_found}")