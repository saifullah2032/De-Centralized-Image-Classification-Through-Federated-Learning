"""
Run Federated Learning Client
Starts a client node for distributed training
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend_fl.fl_client import main

if __name__ == "__main__":
    main()
