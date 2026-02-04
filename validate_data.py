"""
Validation Script for Federated Learning Data Files
Checks model files, history JSON, and database integrity
"""

import os
import json
import sys


def print_status(message, status="info"):
    """Print colored status message"""
    symbols = {
        "success": "[OK]",
        "error": "[ERROR]",
        "warning": "[WARNING]",
        "info": "[INFO]",
    }
    print(f"{symbols.get(status, '')} {message}")


def check_file_exists(filepath, description):
    """Check if a file exists and return its size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_str = (
            f"{size / (1024 * 1024):.2f} MB"
            if size > 1024 * 1024
            else f"{size / 1024:.2f} KB"
        )
        print_status(f"{description} exists ({size_str})", "success")
        return True, size
    else:
        print_status(f"{description} NOT FOUND: {filepath}", "error")
        return False, 0


def validate_model_history():
    """Validate model_history.json structure and data"""
    filepath = "models/model_history.json"

    print("\n" + "=" * 70)
    print("VALIDATING MODEL HISTORY")
    print("=" * 70)

    if not os.path.exists(filepath):
        print_status(f"File not found: {filepath}", "error")
        return False

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        print_status("JSON is valid and parseable", "success")

        # Check structure
        if "rounds" not in data:
            print_status("Missing 'rounds' key in JSON", "error")
            return False

        if "accuracies" not in data:
            print_status("Missing 'accuracies' key in JSON", "error")
            return False

        if "losses" not in data:
            print_status("Missing 'losses' key in JSON", "error")
            return False

        rounds = data["rounds"]
        accuracies = data["accuracies"]
        losses = data["losses"]

        print_status(f"Found {len(rounds)} training rounds", "success")
        print_status(f"Found {len(accuracies)} accuracy values", "success")
        print_status(f"Found {len(losses)} loss values", "success")

        # Check data consistency
        if len(rounds) != len(accuracies) or len(rounds) != len(losses):
            print_status(
                "Data length mismatch between rounds, accuracies, and losses", "error"
            )
            return False

        if rounds:
            # Show first and last round
            print(f"\n  First Round: {rounds[0]}")
            print(f"  - Accuracy: {accuracies[0]:.4f} ({accuracies[0] * 100:.2f}%)")
            print(f"  - Loss: {losses[0]:.4f}")

            print(f"\n  Last Round: {rounds[-1]}")
            print(f"  - Accuracy: {accuracies[-1]:.4f} ({accuracies[-1] * 100:.2f}%)")
            print(f"  - Loss: {losses[-1]:.4f}")

            # Check for valid ranges
            if any(a < 0 or a > 1 for a in accuracies):
                print_status("Some accuracy values are out of range [0, 1]", "warning")

            if any(l < 0 for l in losses):
                print_status("Some loss values are negative", "warning")

        print_status("Model history validation PASSED", "success")
        return True

    except json.JSONDecodeError as e:
        print_status(f"Invalid JSON: {e}", "error")
        return False
    except Exception as e:
        print_status(f"Validation error: {e}", "error")
        return False


def validate_model_files():
    """Validate model files exist"""
    print("\n" + "=" * 70)
    print("VALIDATING MODEL FILES")
    print("=" * 70)

    all_ok = True

    # Check main model
    exists, size = check_file_exists("models/global_model.h5", "Global Model")
    if not exists:
        all_ok = False
    elif size < 1024 * 1024:  # Less than 1MB
        print_status("Model file seems too small, might be corrupted", "warning")

    # Check for checkpoint files
    checkpoint_count = 0
    for i in range(1, 25):  # Check up to 25 rounds
        checkpoint_path = f"models/global_model_round_{i}.h5"
        if os.path.exists(checkpoint_path):
            checkpoint_count += 1

    if checkpoint_count > 0:
        print_status(f"Found {checkpoint_count} checkpoint files", "success")
    else:
        print_status(
            "No checkpoint files found (this is OK if training just started)", "info"
        )

    return all_ok


def validate_database():
    """Validate SQLite database"""
    print("\n" + "=" * 70)
    print("VALIDATING DATABASE")
    print("=" * 70)

    db_path = "app.db"

    if not os.path.exists(db_path):
        print_status(f"Database not found: {db_path}", "warning")
        print_status("Database will be created on first run", "info")
        return True

    exists, size = check_file_exists(db_path, "SQLite Database")

    # Try to connect and query
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if users table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user'"
        )
        if cursor.fetchone():
            print_status("Users table exists", "success")

            # Count users
            cursor.execute("SELECT COUNT(*) FROM user")
            user_count = cursor.fetchone()[0]
            print_status(f"Found {user_count} registered users", "success")
        else:
            print_status("Users table not found", "warning")

        conn.close()
        return True

    except Exception as e:
        print_status(f"Database validation error: {e}", "error")
        return False


def validate_directories():
    """Validate required directories exist"""
    print("\n" + "=" * 70)
    print("VALIDATING DIRECTORIES")
    print("=" * 70)

    required_dirs = [
        ("models", "Model storage"),
        ("uploads", "Temporary uploads"),
        ("logs", "Training logs"),
        ("frontend_web", "Web application"),
        ("backend_fl", "FL backend"),
    ]

    all_ok = True
    for dir_path, description in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print_status(f"{description}: {dir_path}", "success")
        else:
            print_status(f"{description} NOT FOUND: {dir_path}", "error")
            all_ok = False

    return all_ok


def main():
    """Run all validations"""
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING DATA VALIDATION")
    print("=" * 70)

    results = []

    # Run validations
    results.append(("Directories", validate_directories()))
    results.append(("Model Files", validate_model_files()))
    results.append(("Model History", validate_model_history()))
    results.append(("Database", validate_database()))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        status_type = "success" if passed else "error"
        print_status(f"{name}: {status}", status_type)
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print_status("\nAll validations PASSED! System is ready.", "success")
        return 0
    else:
        print_status("\nSome validations FAILED. Check errors above.", "error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
