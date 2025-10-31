import attr

@attr.s
class Container:
    data = attr.ib()

def serializer_using_field(inst, field, value):
    return f"{field.name}={value}"

obj = Container(data=[1, 2, 3])

try:
    result = attr.asdict(obj, recurse=True, value_serializer=serializer_using_field)
    print(f"Result: {result}")
except AttributeError as e:
    print(f"AttributeError occurred: {e}")