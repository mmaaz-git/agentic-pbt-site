import datetime
import random
import string

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
hash_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))