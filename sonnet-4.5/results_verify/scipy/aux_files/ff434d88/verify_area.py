import numpy as np

# Manual calculation of triangle area
# Points forming the convex hull should be [0,0], [1,0], [0,2]
# (The point [0,1] is inside the hull and shouldn't be a vertex)

points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

# Using the shoelace formula for polygon area
def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

area = polygon_area(points)
print(f"Triangle vertices: {points.tolist()}")
print(f"Calculated area using shoelace formula: {area}")

# Alternative: Using cross product for triangle
# Area = 0.5 * |AB x AC|
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([0, 2])

AB = B - A
AC = C - A

# In 2D, cross product gives the z-component (scalar)
cross_product = AB[0] * AC[1] - AB[1] * AC[0]
area_cross = abs(cross_product) / 2
print(f"Area using cross product: {area_cross}")

# Check what vertices the incremental hull actually uses
print("\nVerifying what's happening with incremental hull:")
from scipy.spatial import ConvexHull

# Start with initial 3 points
initial_points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
hull = ConvexHull(initial_points, incremental=True)
print(f"Initial hull vertices indices: {hull.vertices}")
print(f"Initial hull vertex coordinates: {initial_points[hull.vertices]}")
print(f"Initial hull area: {hull.volume}")

# Add new point [0, 2]
new_point = np.array([[0.0, 2.0]])
hull.add_points(new_point)
all_pts = np.vstack([initial_points, new_point])
print(f"\nAfter adding [0, 2]:")
print(f"All points: {all_pts.tolist()}")
print(f"Hull vertices indices: {hull.vertices}")
print(f"Hull vertex coordinates: {all_pts[hull.vertices]}")
print(f"Reported area: {hull.volume}")

# Check if it's calculating area of wrong triangle
# Maybe it's using [0,1], [1,0], [0,2] which would have area 0.5?
wrong_triangle = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0]])
wrong_area = polygon_area(wrong_triangle)
print(f"\nArea if using wrong vertices [0,1], [1,0], [0,2]: {wrong_area}")

# Or maybe averaging old and new?
avg = (0.5 + 1.0) / 2
print(f"Average of old (0.5) and expected new (1.0): {avg}")

# Actually compute what 2/3 would mean
print(f"2/3 = {2/3}")
print(f"Reported incremental volume matches 2/3: {np.isclose(hull.volume, 2/3)}")