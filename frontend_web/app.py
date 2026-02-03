"""
Flask Web Application - Command Center for Federated Learning
Provides web interface for model training monitoring, inference, and administration
"""

import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    Response,
    session,
)
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from flask_cors import CORS

from backend_fl.config import (
    FLASK_SECRET_KEY,
    WEB_HOST,
    WEB_PORT,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    MODEL_HISTORY_PATH,
    PRIVACY_REPORT_PATH,
)
from frontend_web.models import db, init_db, authenticate_user, get_user_by_id
from frontend_web.auth import admin_required
from frontend_web.inference import get_classifier


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = FLASK_SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Configure SQLAlchemy
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'app.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)

# Enable CORS
CORS(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return get_user_by_id(int(user_id))


def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename:
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# PUBLIC ROUTES
# ============================================================================


@app.route("/")
def index():
    """Landing page"""
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = authenticate_user(username, password)

        if user:
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")

            # Redirect based on role
            if user.is_admin():
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("predict"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration page"""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        password_confirm = request.form.get("password_confirm")
        role = request.form.get("role", "client")

        # Validation
        if not username or not password or not password_confirm:
            flash("Please fill in all fields.", "danger")
        elif len(username) < 3:
            flash("Username must be at least 3 characters long.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
        elif password != password_confirm:
            flash("Passwords do not match.", "danger")
        else:
            # Check if user already exists
            from frontend_web.models import get_user_by_username, User as UserModel

            existing_user = get_user_by_username(username)
            if existing_user:
                flash(f"Username '{username}' already exists.", "danger")
            else:
                # Create new user
                try:
                    new_user = UserModel(
                        username=username, password=password, role=role
                    )
                    db.session.add(new_user)
                    db.session.commit()
                    flash(
                        f"Registration successful! Please log in as {username}.",
                        "success",
                    )
                    return redirect(url_for("login"))
                except Exception as e:
                    db.session.rollback()
                    flash(f"Registration failed: {str(e)}", "danger")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    """Logout"""
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ============================================================================
# PREDICTION ROUTES
# ============================================================================


@app.route("/predict", methods=["GET"])
@login_required
def predict():
    """Prediction page"""
    classifier = get_classifier()
    model_info = classifier.get_model_info()
    return render_template("predict.html", model_info=model_info)


@app.route("/predict", methods=["POST"])
@login_required
def predict_image():
    """Handle image upload and prediction"""
    import base64

    # Check if file is in request
    if "file" not in request.files:
        flash("No file provided", "danger")
        return redirect(url_for("predict"))

    file = request.files["file"]

    # Check if file is selected
    if file.filename == "":
        flash("No file selected", "danger")
        return redirect(url_for("predict"))

    # Check if file is allowed
    if not allowed_file(file.filename):
        flash(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "danger")
        return redirect(url_for("predict"))

    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Make prediction
        classifier = get_classifier()
        result = classifier.predict(filepath)

        # Read image for display (convert to base64)
        image_data = ""
        try:
            with open(filepath, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode()
        except:
            pass

        # Delete uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        if not result.get("success"):
            flash(
                f"Prediction failed: {result.get('error', 'Unknown error')}", "danger"
            )
            return redirect(url_for("predict"))

        # Get model info for display
        model_info = classifier.get_model_info()

        # Get current accuracy from model history
        current_accuracy = 0.0
        rounds_completed = 0
        if os.path.exists(MODEL_HISTORY_PATH):
            try:
                with open(MODEL_HISTORY_PATH, "r") as f:
                    history = json.load(f)
                    if history.get("rounds"):
                        rounds_completed = history["rounds"][-1]
                        current_accuracy = history["accuracies"][-1] * 100
            except:
                pass

        # Prepare data for template
        template_data = {
            "image_data": image_data,
            "predicted_class": result["predicted_class"],
            "confidence_percent": int(result["confidence"] * 100),
            "probabilities": [
                (p["class_name"], p["probability"]) for p in result["all_predictions"]
            ],
            "model_arch": model_info.get("model_path", "Unknown"),
            "model_params": f"{model_info.get('total_params', 0):,}"
            if "total_params" in model_info
            else "Unknown",
            "rounds_completed": rounds_completed,
            "current_accuracy": f"{current_accuracy:.2f}",
        }

        return render_template("results.html", **template_data)

    except Exception as e:
        flash(f"Error processing image: {str(e)}", "danger")
        return redirect(url_for("predict"))


# ============================================================================
# API ROUTES
# ============================================================================


@app.route("/api/status")
def api_status():
    """Get system status"""
    classifier = get_classifier()
    model_info = classifier.get_model_info()

    # Get model history if available
    current_round = 0
    accuracy = 0.0

    if os.path.exists(MODEL_HISTORY_PATH):
        try:
            with open(MODEL_HISTORY_PATH, "r") as f:
                history = json.load(f)
                if history["rounds"]:
                    current_round = history["rounds"][-1]
                    accuracy = history["accuracies"][-1]
        except:
            pass

    return jsonify(
        {
            "model_loaded": model_info["model_loaded"],
            "current_round": current_round,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/metrics")
def api_metrics():
    """Get training metrics"""
    if not os.path.exists(MODEL_HISTORY_PATH):
        return jsonify(
            {"success": False, "error": "No training history available"}
        ), 404

    try:
        with open(MODEL_HISTORY_PATH, "r") as f:
            history = json.load(f)
        return jsonify({"success": True, "data": history})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# ADMIN ROUTES
# ============================================================================


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    return render_template("admin_dashboard.html")


@app.route("/client/dashboard")
@login_required
def client_dashboard():
    """Client dashboard - for clients to monitor local training"""
    if current_user.is_admin():
        flash("Admin users should use the admin dashboard.", "info")
        return redirect(url_for("admin_dashboard"))
    return render_template("client_dashboard.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """General dashboard - redirects based on role"""
    if current_user.is_admin():
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("client_dashboard"))


@app.route("/admin/events")
@admin_required
def admin_events():
    """Server-Sent Events stream for real-time updates"""

    def generate():
        """Generate SSE events"""
        import time
        from backend_fl.training_status import get_training_status
        from backend_fl.config import TRAINING_LOG_PATH

        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Connected to event stream', 'timestamp': datetime.now().isoformat()})}\n\n"

        last_position = 0
        last_status = None

        if os.path.exists(TRAINING_LOG_PATH):
            with open(TRAINING_LOG_PATH, "r") as f:
                f.seek(0, 2)  # Go to end of file
                last_position = f.tell()

        while True:
            # Check for training status updates
            current_status = get_training_status()
            if current_status != last_status:
                last_status = current_status
                yield f"data: {json.dumps({'type': 'status', 'data': current_status})}\n\n"

            # Check for new log entries
            if os.path.exists(TRAINING_LOG_PATH):
                with open(TRAINING_LOG_PATH, "r") as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()

                    for line in new_lines:
                        if line.strip():
                            event_data = {
                                "type": "log",
                                "message": line.strip(),
                                "timestamp": datetime.now().isoformat(),
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"

            # Send heartbeat
            time.sleep(2)
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/privacy-report")
@login_required
def privacy_report():
    """Display privacy verification report"""
    if not os.path.exists(PRIVACY_REPORT_PATH):
        return render_template("privacy_report.html", report=None)

    try:
        with open(PRIVACY_REPORT_PATH, "r") as f:
            report_data = json.load(f)
        return render_template("privacy_report.html", report=report_data)
    except Exception as e:
        flash(f"Error loading privacy report: {e}", "danger")
        return render_template("privacy_report.html", report=None)


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template("500.html"), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Initialize database
    init_db(app)

    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING COMMAND CENTER")
    print("=" * 70)
    print(f"  Host:     {WEB_HOST}")
    print(f"  Port:     {WEB_PORT}")
    print(f"  Database: SQLite (persistent)")
    print(f"  URL:      http://localhost:{WEB_PORT}")
    print("=" * 70)
    print(f"  Default Credentials:")
    print(f"    Admin:  admin / admin123")
    print(f"    Client: client / client123")
    print("=" * 70 + "\n")

    # Run Flask app
    app.run(host=WEB_HOST, port=WEB_PORT, debug=True)
