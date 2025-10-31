import numpy as np
from scipy.spatial import Delaunay

# Test the specific failing case: n_points=13, seed=640
n_points = 13
seed = 640

np.random.seed(seed)
points = np.random.randn(n_points, 2)

tri = Delaunay(points)

print(f"Testing {n_points} points with seed {seed}")
print(f"Points shape: {points.shape}")

failed_points = []
for i in range(len(points)):
    simplex_idx = tri.find_simplex(points[i])
    if simplex_idx < 0:
        failed_points.append((i, points[i], simplex_idx))
        print(f"FAILED: Point {i} at {points[i]} not found in any simplex (returned {simplex_idx})")
    else:
        print(f"OK: Point {i} found in simplex {simplex_idx}")

if failed_points:
    print(f"\n{len(failed_points)} points failed out of {len(points)}")
else:
    print("\nAll points found successfully!")