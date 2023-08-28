import os
from datetime import datetime

import yaml

# Default configuration values
class DEFAULT:
    class IP:
        host = "192.168.0.8"
        hololens = "192.168.0.2"

    class PORT:
        host = 10024

    class AUDIO:
        ip = "192.168.0.8"
        port = 28000
        buffer_size = 1024

# Constants
CONFIG = ".config.yaml"
CURDIR = os.path.dirname(os.path.abspath(__file__))
SAVEDIR = os.path.join(CURDIR, "data", datetime.now().strftime("%Y-%m-%d_%H%M%S"))

# Function to clear the console
def clear():
    os.system("cls" if os.name == "nt" else "clear")


# Load configuration from a YAML file
def load_config(filename):
    with open(filename, encoding="utf-8") as f:
        return yaml.safe_load(f)


