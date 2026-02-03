"""
Database Models Module
Implements SQLAlchemy ORM models for persistent data storage
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import os

# Initialize SQLAlchemy
db = SQLAlchemy()


class User(db.Model, UserMixin):
    """
    User model for database-backed authentication
    Supports roles: admin, client
    """

    __tablename__ = "users"

    # Columns
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default="client", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    def __init__(self, username, password, role="client"):
        self.username = username
        self.password_hash = generate_password_hash(password)
        self.role = role
        self.created_at = datetime.utcnow()

    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        """Check if user has admin role"""
        return self.role == "admin"

    def is_client(self):
        """Check if user has client role"""
        return self.role == "client"

    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"

    def to_dict(self):
        """Convert user to dictionary"""
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
        }


class TrainingSession(db.Model):
    """
    Model to track federated learning training sessions
    Useful for logging and monitoring
    """

    __tablename__ = "training_sessions"

    id = db.Column(db.Integer, primary_key=True)
    session_name = db.Column(db.String(255), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time = db.Column(db.DateTime)
    status = db.Column(
        db.String(50), default="running", nullable=False
    )  # running, completed, failed
    total_rounds = db.Column(db.Integer)
    completed_rounds = db.Column(db.Integer, default=0)
    final_accuracy = db.Column(db.Float)
    final_loss = db.Column(db.Float)
    notes = db.Column(db.Text)

    def __repr__(self):
        return f"<TrainingSession {self.session_name} ({self.status})>"

    def to_dict(self):
        """Convert session to dictionary"""
        return {
            "id": self.id,
            "session_name": self.session_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "total_rounds": self.total_rounds,
            "completed_rounds": self.completed_rounds,
            "final_accuracy": self.final_accuracy,
            "final_loss": self.final_loss,
            "notes": self.notes,
        }


def init_db(app):
    """
    Initialize database with Flask app
    Creates tables and adds default users if needed

    Args:
        app: Flask application instance
    """
    with app.app_context():
        # Create all tables
        db.create_all()

        # Add default admin user if none exists
        admin_exists = User.query.filter_by(username="admin").first()
        if not admin_exists:
            admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
            admin_user = User(username="admin", password=admin_password, role="admin")
            db.session.add(admin_user)
            print(
                "[OK] Created default admin user (username: admin, password: admin123)"
            )

        # Add default client user if none exists
        client_exists = User.query.filter_by(username="client").first()
        if not client_exists:
            client_user = User(username="client", password="client123", role="client")
            db.session.add(client_user)
            print(
                "[OK] Created default client user (username: client, password: client123)"
            )

        try:
            db.session.commit()
            print("[OK] Database initialized successfully")
        except Exception as e:
            print(f"[!] Database initialization warning: {e}")
            db.session.rollback()


def create_user(username, password, role="client"):
    """
    Create a new user in the database

    Args:
        username: Username
        password: Plain text password
        role: User role ('admin' or 'client')

    Returns:
        User object if successful, None if user already exists
    """
    if User.query.filter_by(username=username).first():
        return None  # User already exists

    user = User(username=username, password=password, role=role)
    db.session.add(user)
    db.session.commit()
    return user


def get_user_by_id(user_id):
    """Get user by ID"""
    return User.query.get(user_id)


def get_user_by_username(username):
    """Get user by username"""
    return User.query.filter_by(username=username).first()


def authenticate_user(username, password):
    """
    Authenticate user with username and password

    Args:
        username: Username
        password: Plain text password

    Returns:
        User object if successful, None otherwise
    """
    user = get_user_by_username(username)
    if user and user.is_active and user.check_password(password):
        user.update_last_login()
        return user
    return None


def get_all_users():
    """Get all users"""
    return User.query.all()


def delete_user(user_id):
    """Delete user by ID"""
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
        return True
    return False


def update_user_role(user_id, new_role):
    """Update user role"""
    user = User.query.get(user_id)
    if user:
        user.role = new_role
        db.session.commit()
        return True
    return False
