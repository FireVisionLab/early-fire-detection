import sys
from pathlib import Path

__ROOT = str(Path(__file__).resolve().parent.parent) 
if __ROOT not in sys.path:
    sys.path.insert(0, __ROOT)  
