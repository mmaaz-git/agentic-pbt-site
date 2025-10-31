import numpy.rec
import traceback

try:
    parser = numpy.rec.format_parser(['i4', 'i4'], [0, 1], [])
    print("No error raised - this is unexpected!")
except AttributeError as e:
    print(f"Got AttributeError: {e}")
    print("\nFull traceback:")
    traceback.print_exc()