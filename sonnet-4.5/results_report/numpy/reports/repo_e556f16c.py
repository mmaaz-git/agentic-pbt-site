import numpy.rec

# This demonstrates the bug where format_parser crashes with an unhelpful
# AttributeError when given non-string field names, instead of raising
# a clear TypeError or ValueError explaining the input requirement.

try:
    parser = numpy.rec.format_parser(['i4', 'i4'], [0, 1], [])
    print("No error raised - this should not happen!")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

    # Show full traceback
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()