import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.plugin._schema_validator import PluggableSchemaValidator
from pydantic.plugin import SchemaTypePath


class BadPlugin:
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        # Return wrong number of elements (2 instead of 3)
        return (None, None)


schema = {'type': 'int'}
schema_type = int
schema_type_path = SchemaTypePath('test', 'test')
schema_kind = 'TypeAdapter'

try:
    validator = PluggableSchemaValidator(
        schema, schema_type, schema_type_path, schema_kind, None, [BadPlugin()], {}
    )
    print("ERROR: No exception was raised!")
except ValueError as e:
    print(f"ValueError: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Does error identify plugin? {'BadPlugin' in str(e) or 'Error using plugin' in str(e)}")
except TypeError as e:
    print(f"TypeError: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Does error identify plugin? {'BadPlugin' in str(e) or 'Error using plugin' in str(e)}")