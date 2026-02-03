"""
Run Federated Learning Server
Starts the FL server to coordinate training across distributed clients
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend_fl.fl_server import main

if __name__ == "__main__":
    main()
