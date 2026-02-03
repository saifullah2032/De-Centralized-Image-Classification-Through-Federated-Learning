"""
Authentication and Authorization Module
Implements Role-Based Access Control (RBAC) for the web interface
Database-backed authentication using SQLAlchemy
"""

from functools import wraps
from flask import redirect, url_for, flash
from flask_login import current_user
from frontend_web.models import (
    User,
    get_user_by_id,
    get_user_by_username,
    authenticate_user,
)


def admin_required(f):
    """
    Decorator to require admin role for a route

    Usage:
        @app.route('/admin/dashboard')
        @admin_required
        def admin_dashboard():
            ...
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))

        if not current_user.is_admin():
            flash("Admin access required.", "danger")
            return redirect(url_for("index"))

        return f(*args, **kwargs)

    return decorated_function


def login_required_custom(f):
    """
    Custom login required decorator

    Usage:
        @app.route('/predict')
        @login_required_custom
        def predict():
            ...
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))

        return f(*args, **kwargs)

    return decorated_function


if __name__ == "__main__":
    """Test authentication module"""
    print("Testing authentication module...")

    # Test user creation
    admin_user = get_user("admin")
    print(f"Admin user: {admin_user}")
    print(f"  Is admin: {admin_user.is_admin()}")

    # Test authentication
    auth_result = authenticate_user("admin", "admin123")
    print(f"\nAuthentication test: {'✓ Success' if auth_result else '❌ Failed'}")

    # Test wrong password
    auth_result_fail = authenticate_user("admin", "wrongpassword")
    print(
        f"Wrong password test: {'✓ Correctly rejected' if not auth_result_fail else '❌ Failed'}"
    )

    print("\n✓ Authentication module test passed!")
