import numpy.rec

rec_arr = numpy.rec.fromarrays([[0, 0]], names='x')
print(f"Number of fields: {len(rec_arr.dtype.names)}")

# This works - field index 0 exists
result = rec_arr.field(0)
print(f"rec_arr.field(0) returned: {result}")

# This should raise an error - field index 1 doesn't exist
try:
    rec_arr.field(1)
except IndexError as e:
    print(f"Error: {e}")