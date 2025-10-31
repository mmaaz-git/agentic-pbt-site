import numpy.rec as rec

names = ['\r']
formats = ['i4']

parser = rec.format_parser(formats, names, [])

print(f"Input name: {repr(names[0])}")
print(f"Actual name: {repr(parser.dtype.names[0])}")

try:
    field = parser.dtype.fields['\r']
    print("Can access with '\\r': SUCCESS")
except KeyError:
    print("Cannot access with '\\r': FAILED")

try:
    field = parser.dtype.fields['']
    print("Can access with '': SUCCESS")
except KeyError:
    print("Cannot access with '': FAILED")