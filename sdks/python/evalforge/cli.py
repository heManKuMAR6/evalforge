import subprocess
import sys
from .client import _find_binary

def main():
    binary = _find_binary()
    result = subprocess.run([binary] + sys.argv[1:])
    sys.exit(result.returncode)
