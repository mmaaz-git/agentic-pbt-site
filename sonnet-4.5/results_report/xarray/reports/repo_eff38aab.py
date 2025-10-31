import numpy as np
from xarray.indexes.range_index import RangeCoordinateTransform

transform = RangeCoordinateTransform(
    start=0.0,
    stop=0.0,
    size=1,
    coord_name="x",
    dim="x",
)

print(f"step: {transform.step}")

original_labels = {transform.coord_name: np.array([0.0])}
positions = transform.reverse(original_labels)
reconstructed_labels = transform.forward(positions)

print(f"Original labels: {original_labels[transform.coord_name]}")
print(f"After round-trip: {reconstructed_labels[transform.coord_name]}")