import os
import sys
first_path = sys.path[0]
parent_path = os.path.dirname(first_path)
sys.path.insert(0, parent_path)
