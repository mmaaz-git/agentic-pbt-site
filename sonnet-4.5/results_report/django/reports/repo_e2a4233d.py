from django.core.serializers.base import deserialize_m2m_values


class ErrorIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise RuntimeError("Error during iteration")


class MockPK:
    def to_python(self, v):
        return int(v)


class MockMeta:
    pk = MockPK()


class MockDefaultManager:
    pass


class MockModel:
    _meta = MockMeta()
    _default_manager = MockDefaultManager()


class MockRemoteField:
    model = MockModel


class MockField:
    remote_field = MockRemoteField()


field = MockField()
field_value = ErrorIterator()

try:
    result = deserialize_m2m_values(field, field_value, using='default', handle_forward_references=False)
    print("No error occurred (unexpected)")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    import traceback
    traceback.print_exc()