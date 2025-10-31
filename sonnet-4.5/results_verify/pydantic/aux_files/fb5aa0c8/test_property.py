import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from hypothesis import given, strategies as st, settings, assume
from pydantic.plugin._schema_validator import PluggableSchemaValidator
from pydantic.plugin import SchemaTypePath


@given(st.integers(min_value=0, max_value=10))
@settings(max_examples=50)
def test_plugin_wrong_tuple_size_error_message(tuple_size):
    assume(tuple_size != 3)

    class BadPlugin:
        def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
            return tuple([None] * tuple_size)

    schema = {'type': 'int'}
    schema_type = int
    schema_type_path = SchemaTypePath('test', 'test')
    schema_kind = 'TypeAdapter'

    try:
        validator = PluggableSchemaValidator(
            schema, schema_type, schema_type_path, schema_kind, None, [BadPlugin()], {}
        )
        raise AssertionError(f"No exception raised for tuple size {tuple_size}")
    except (TypeError, ValueError) as e:
        error_msg = str(e)
        has_plugin_info = ("BadPlugin" in error_msg or
                          "Error using plugin" in error_msg or
                          "test_valueerror_bug" in error_msg)
        if not has_plugin_info:
            raise AssertionError(
                f"BUG: Exception doesn't identify which plugin failed!\n"
                f"Tuple size: {tuple_size}\n"
                f"Exception type: {type(e).__name__}\n"
                f"Message: {error_msg}"
            )

if __name__ == "__main__":
    test_plugin_wrong_tuple_size_error_message()
    print("Test completed")