"""Run this experiment from the repository root. See the adjacent README."""
from src.reproduce import main
import sys
if __name__ == '__main__':
    main(['trajectories', *sys.argv[1:]])
