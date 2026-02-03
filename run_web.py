"""
Run Flask Web Application
Starts the Command Center web interface
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend_web.app import app, WEB_HOST, WEB_PORT, UPLOAD_FOLDER

if __name__ == "__main__":
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING COMMAND CENTER")
    print("=" * 70)
    print(f"  Host:     {WEB_HOST}")
    print(f"  Port:     {WEB_PORT}")
    print(f"  Login:    admin / admin123 (default credentials)")
    print(f"  URL:      http://localhost:{WEB_PORT}")
    print("=" * 70 + "\n")

    # Run Flask app
    app.run(host=WEB_HOST, port=WEB_PORT, debug=True)
