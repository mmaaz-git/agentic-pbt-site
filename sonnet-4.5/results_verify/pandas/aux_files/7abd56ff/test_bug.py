from pandas.io.formats.printing import _justify

# Test case from bug report
head = [['a', 'b', 'c']]
tail = [['x']]
result_head, result_tail = _justify(head, tail)

print(f"Input:  head={head}, tail={tail}")
print(f"Output: head={result_head}, tail={result_tail}")
print(f"Expected: head should have 3 elements, tail should have 1")
print(f"Actual:   head has {len(result_head[0])} elements")
print()

# Also test the other cases mentioned
head2 = [['', '']]
tail2 = [['']]
result_head2, result_tail2 = _justify(head2, tail2)
print(f"Test 2: head=[['', '']], tail=[['']]")
print(f"Result: head={result_head2}, tail={result_tail2}")
print(f"Expected head length: 2, Actual: {len(result_head2[0])}")
print()

head3 = [['']]
tail3 = [['', '']]
result_head3, result_tail3 = _justify(head3, tail3)
print(f"Test 3: head=[['']], tail=[['', '']]")
print(f"Result: head={result_head3}, tail={result_tail3}")
print(f"Expected tail length: 2, Actual: {len(result_tail3[0])}")