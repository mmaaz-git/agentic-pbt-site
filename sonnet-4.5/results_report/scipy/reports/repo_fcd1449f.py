import numpy as np
from scipy.spatial import ConvexHull

# Initial points and new point from the bug report
points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
new_point = np.array([[0.0, 2.0]])

# Incremental construction
hull_incremental = ConvexHull(points, incremental=True)
print(f"Initial hull volume: {hull_incremental.volume}")
print(f"Initial hull vertices: {points[hull_incremental.vertices]}")
hull_incremental.add_points(new_point)
print(f"Incremental volume after add_points: {hull_incremental.volume}")
print(f"Incremental vertices after add_points: {hull_incremental.points[hull_incremental.vertices]}")

# Batch construction with all points
all_points = np.vstack([points, new_point])
hull_batch = ConvexHull(all_points)
print(f"Batch volume: {hull_batch.volume}")
print(f"Batch vertices: {hull_batch.points[hull_batch.vertices]}")

# Verification
print(f"\nVerification:")
print(f"Incremental and batch volumes match: {np.isclose(hull_incremental.volume, hull_batch.volume)}")
print(f"Volume difference: {abs(hull_incremental.volume - hull_batch.volume)}")
print(f"Percentage error: {abs(hull_incremental.volume - hull_batch.volume) / hull_batch.volume * 100:.1f}%")

# Manual calculation for triangle with vertices [0,0], [1,0], [0,2]
# Area = 0.5 * base * height = 0.5 * 1 * 2 = 1.0
print(f"\nManual calculation for triangle [0,0], [1,0], [0,2]:")
print(f"Expected area: 1.0 (0.5 * base(1) * height(2))")