import numpy as np
from scipy.spatial import ConvexHull

# Test the specific failing input
points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
new_point = np.array([[0.0, 2.0]])

print("Testing specific failing input:")
print(f"Initial points: {points.tolist()}")
print(f"New point: {new_point.tolist()}")

# Incremental construction
hull_incremental = ConvexHull(points, incremental=True)
print(f"\nInitial hull volume: {hull_incremental.volume}")
print(f"Initial hull vertices: {sorted(hull_incremental.vertices.tolist())}")

hull_incremental.add_points(new_point)
print(f"\nAfter add_points:")
print(f"Incremental hull volume: {hull_incremental.volume}")
print(f"Incremental hull vertices: {sorted(hull_incremental.vertices.tolist())}")

# Batch construction with all points
all_points = np.vstack([points, new_point])
hull_batch = ConvexHull(all_points)
print(f"\nBatch hull volume: {hull_batch.volume}")
print(f"Batch hull vertices: {sorted(hull_batch.vertices.tolist())}")

# Compare
print(f"\nVolume difference: {abs(hull_incremental.volume - hull_batch.volume)}")
print(f"Volumes match: {np.isclose(hull_incremental.volume, hull_batch.volume)}")

# Manual calculation of expected area (2D convex hull "volume" is area)
# Triangle with vertices at [0,0], [1,0], [0,2]
# Area = 0.5 * base * height = 0.5 * 1 * 2 = 1.0
print(f"\nExpected area (manual calculation): 1.0")

# Additional checks
print(f"\nIncremental hull points shape: {hull_incremental.points.shape}")
print(f"Batch hull points shape: {hull_batch.points.shape}")
print(f"Incremental hull simplices:\n{hull_incremental.simplices}")
print(f"Batch hull simplices:\n{hull_batch.simplices}")