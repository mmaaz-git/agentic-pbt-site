import os
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/flask_env/lib/python3.13/site-packages')

from flask.helpers import get_debug_flag, get_load_dotenv

print("Testing get_debug_flag():")
print("-" * 40)

os.environ["FLASK_DEBUG"] = "false"
print(f"FLASK_DEBUG='false' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = " false "
print(f"FLASK_DEBUG=' false ' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = "  "
print(f"FLASK_DEBUG='  ' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = "\tfalse\t"
print(f"FLASK_DEBUG='\\tfalse\\t' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = "0"
print(f"FLASK_DEBUG='0' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = " 0 "
print(f"FLASK_DEBUG=' 0 ' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = "no"
print(f"FLASK_DEBUG='no' → {get_debug_flag()}")

os.environ["FLASK_DEBUG"] = " no "
print(f"FLASK_DEBUG=' no ' → {get_debug_flag()}")

print("\nTesting get_load_dotenv():")
print("-" * 40)

os.environ["FLASK_SKIP_DOTENV"] = "no"
print(f"FLASK_SKIP_DOTENV='no' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = " no "
print(f"FLASK_SKIP_DOTENV=' no ' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = "  "
print(f"FLASK_SKIP_DOTENV='  ' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = "\tno\t"
print(f"FLASK_SKIP_DOTENV='\\tno\\t' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = "false"
print(f"FLASK_SKIP_DOTENV='false' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = " false "
print(f"FLASK_SKIP_DOTENV=' false ' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = "0"
print(f"FLASK_SKIP_DOTENV='0' → {get_load_dotenv()}")

os.environ["FLASK_SKIP_DOTENV"] = " 0 "
print(f"FLASK_SKIP_DOTENV=' 0 ' → {get_load_dotenv()}")