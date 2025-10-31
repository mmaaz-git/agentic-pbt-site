import numpy as np
from scipy.spatial import Delaunay

np.random.seed(640)
points = np.random.randn(13, 2)

tri = Delaunay(points)
point_1 = points[1]

# Check if point 1 is actually a vertex in the triangulation
print("Checking if point 1 is a vertex in the triangulation:")
print(f"Point 1 coordinates: {point_1}")

# Check if it appears in tri.points (the actual vertices used)
print(f"\ntri.points[1]: {tri.points[1]}")
print(f"Are they the same? {np.array_equal(point_1, tri.points[1])}")
print(f"Difference: {point_1 - tri.points[1]}")

# Check in simplices
print(f"\nPoint 1 appears in simplices: ", end="")
simplices_with_point_1 = []
for s_idx, simplex in enumerate(tri.simplices):
    if 1 in simplex:
        simplices_with_point_1.append(s_idx)
        print(s_idx, end=" ")
print()

# Verify that it's really a vertex
print(f"\nTotal simplices containing vertex 1: {len(simplices_with_point_1)}")

# Check coplanar points
print(f"\nCoplanar points: {tri.coplanar}")
print(f"Is point 1 in coplanar list? {1 in tri.coplanar if tri.coplanar.size > 0 else False}")