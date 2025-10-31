import numpy.rec

# Test case 1: field name 'field'
rec = numpy.rec.fromrecords([(1,), (2,), (3,)], names='field')

print("Test with field name 'field':")
print("rec['field'] =", rec['field'])
print("type(rec.field) =", type(rec.field))
print("rec.field =", rec.field)

try:
    value = rec.field[0]
    print("rec.field[0] =", value)
except TypeError as e:
    print(f"TypeError when accessing rec.field[0]: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: field name 'copy'
rec2 = numpy.rec.fromrecords([(10,), (20,), (30,)], names='copy')
print("Test with field name 'copy':")
print("rec2['copy'] =", rec2['copy'])
print("type(rec2.copy) =", type(rec2.copy))
print("rec2.copy =", rec2.copy)

print("\n" + "="*50 + "\n")

# Test case 3: field name 'sum'
rec3 = numpy.rec.fromrecords([(100,), (200,), (300,)], names='sum')
print("Test with field name 'sum':")
print("rec3['sum'] =", rec3['sum'])
print("type(rec3.sum) =", type(rec3.sum))
print("rec3.sum =", rec3.sum)

print("\n" + "="*50 + "\n")

# Test case 4: non-conflicting field name
rec4 = numpy.rec.fromrecords([(1000,), (2000,), (3000,)], names='myfield')
print("Test with field name 'myfield' (non-conflicting):")
print("rec4['myfield'] =", rec4['myfield'])
print("type(rec4.myfield) =", type(rec4.myfield))
print("rec4.myfield =", rec4.myfield)
print("rec4.myfield[0] =", rec4.myfield[0])