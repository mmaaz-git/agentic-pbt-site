import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.plugin._schema_validator import PluggableSchemaValidator
from pydantic.plugin import SchemaTypePath


class BadPlugin:
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        return (None, None)  # Returns 2 elements instead of 3


schema = {'type': 'int'}
schema_type = int
schema_type_path = SchemaTypePath('test', 'test')
schema_kind = 'TypeAdapter'

try:
    validator = PluggableSchemaValidator(
        schema, schema_type, schema_type_path, schema_kind, None, [BadPlugin()], {}
    )
except ValueError as e:
    print(f"ValueError caught: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Does error identify plugin? {'BadPlugin' in str(e) or 'Error using plugin' in str(e)}")
except TypeError as e:
    print(f"TypeError caught: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Does error identify plugin? {'BadPlugin' in str(e) or 'Error using plugin' in str(e)}")