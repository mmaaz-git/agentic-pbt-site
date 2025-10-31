import math

# Manual calculation of what happens in RangeIndex.arange
start = -1.5
stop = 0.0
step = 1.0

# Line 219: size = math.ceil((stop - start) / step)
size = math.ceil((stop - start) / step)
print(f"Calculated size: math.ceil(({stop} - {start}) / {step}) = math.ceil({(stop - start) / step}) = {size}")

# Lines 221-223: Create RangeCoordinateTransform without passing step
# Then in line 62: recalculated step = (self.stop - self.start) / self.size
recalculated_step = (stop - start) / size
print(f"Recalculated step: ({stop} - {start}) / {size} = {recalculated_step}")

print()
print("Problem:")
print(f"  Original step: {step}")
print(f"  Recalculated step: {recalculated_step}")
print(f"  These are different!")

print()
print("Values generated:")
print("  With original step (NumPy):")
values_numpy = []
val = start
while val < stop:
    values_numpy.append(val)
    val += step
print(f"    {values_numpy}")

print("  With recalculated step (RangeIndex):")
values_xarray = []
for i in range(size):
    values_xarray.append(start + i * recalculated_step)
print(f"    {values_xarray}")