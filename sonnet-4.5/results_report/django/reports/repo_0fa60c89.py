from django.db.migrations.serializer import DictionarySerializer

# Test case with mixed-type dictionary keys
d = {0: 0, '': 0}
serializer = DictionarySerializer(d)

try:
    result, imports = serializer.serialize()
    print(f"Success: {result}")
    print(f"Imports: {imports}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()