import numpy.char as char

s = '\x00'

functions = ['upper', 'lower', 'capitalize', 'title', 'swapcase', 'strip']
for func_name in functions:
    numpy_func = getattr(char, func_name)
    python_func = getattr(str, func_name)

    numpy_result = numpy_func(s).item()
    python_result = python_func(s)

    print(f"{func_name}({repr(s)}): numpy={repr(numpy_result)}, python={repr(python_result)}")
    if numpy_result != python_result:
        print(f"  -> MISMATCH!")
    else:
        print(f"  -> OK")

# Test additional functions mentioned
print("\nAdditional functions:")

# Test lstrip
numpy_result = char.lstrip('\x00').item()
python_result = str.lstrip('\x00')
print(f"lstrip({repr('\x00')}): numpy={repr(numpy_result)}, python={repr(python_result)}")
if numpy_result != python_result:
    print(f"  -> MISMATCH!")
else:
    print(f"  -> OK")

# Test rstrip
numpy_result = char.rstrip('\x00').item()
python_result = str.rstrip('\x00')
print(f"rstrip({repr('\x00')}): numpy={repr(numpy_result)}, python={repr(python_result)}")
if numpy_result != python_result:
    print(f"  -> MISMATCH!")
else:
    print(f"  -> OK")

# Test encode
numpy_result = char.encode('\x00', encoding='utf-8').item()
python_result = '\x00'.encode('utf-8')
print(f"encode({repr('\x00')}): numpy={repr(numpy_result)}, python={repr(python_result)}")
if numpy_result != python_result:
    print(f"  -> MISMATCH!")
else:
    print(f"  -> OK")