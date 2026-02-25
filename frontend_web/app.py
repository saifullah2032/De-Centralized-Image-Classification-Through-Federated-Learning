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
from frontend_web.inference import get_classifier, switch_model, get_available_models


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


@app.route("/favicon.ico")
def favicon():
    """Handle favicon requests to avoid 404s"""
    return "", 204


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
    available_models = get_available_models()
    return render_template(
        "predict.html",
        model_info=model_info,
        available_models=available_models,
        current_model=classifier.current_model_type,
    )


@app.route("/results")
@login_required
def show_results():
    """Display prediction results"""
    # Get results from session
    results = session.get("prediction_results")

    if not results:
        flash("No prediction results found. Please upload an image first.", "warning")
        return redirect(url_for("predict"))

    return render_template("results.html", **results)


@app.route("/temp_image/<filename>")
@login_required
def serve_temp_image(filename):
    """Serve temporary uploaded image"""
    from flask import send_from_directory

    app.logger.info(f"Request for temp image: {filename}")

    # Security: only serve if this filename is in the user's session
    results = session.get("prediction_results", {})
    session_filename = results.get("temp_image_file")

    app.logger.info(f"Session has filename: {session_filename}")

    if session_filename != filename:
        app.logger.warning(f"Unauthorized access attempt - session mismatch")
        return "Unauthorized", 403

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    app.logger.info(f"Looking for file at: {filepath}")

    if not os.path.exists(filepath):
        app.logger.error(f"File not found: {filepath}")
        return "Image not found", 404

    app.logger.info(f"Serving image: {filename}")
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
@login_required
def predict_image():
    """Handle image upload and prediction"""
    import base64
    import io
    from PIL import Image

    # Clear old prediction results from session
    session.pop("prediction_results", None)

    # Check if file is in request (upload method)
    filepath = None
    filename = None

    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
    elif request.form.get("file"):
        # Handle base64 captured image
        try:
            img_data = request.form.get("file")
            if img_data.startswith("data:image"):
                img_data = img_data.split(",")[1]

            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_captured.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img.save(filepath, "JPEG")

            file = None
        except Exception as e:
            app.logger.error(f"Error processing captured image: {str(e)}")
            flash(f"Error processing captured image: {str(e)}", "danger")
            return redirect(url_for("predict"))
    else:
        app.logger.warning("No file in request")
        flash("No image provided. Please upload or capture an image.", "danger")
        return redirect(url_for("predict"))

    try:
        # Handle file upload method
        if file:
            # Check if file is selected
            if file.filename == "":
                app.logger.warning("Empty filename")
                flash("No file selected. Please choose an image.", "danger")
                return redirect(url_for("predict"))

            # Check if file is allowed
            if not allowed_file(file.filename):
                app.logger.warning(f"Invalid file type: {file.filename}")
                flash(
                    f"Invalid file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}",
                    "danger",
                )
                return redirect(url_for("predict"))

            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            app.logger.info(f"Saving uploaded file to: {filepath}")
            file.save(filepath)

            # Verify file was saved
            if not os.path.exists(filepath):
                raise Exception(f"Failed to save file to {filepath}")

        # Make prediction (this is the slow part - 20-30s on first call)
        app.logger.info(f"Loading classifier for prediction...")
        classifier = get_classifier()

        app.logger.info(f"Running inference on {filename}...")
        result = classifier.predict(filepath)

        # Save image temporarily (don't delete yet, needed for results page)
        # We'll store the filename in session instead of base64 data
        temp_image_filename = filename

        # Check if prediction was successful
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error occurred during prediction")
            app.logger.error(f"Prediction failed: {error_msg}")
            flash(f"Prediction failed: {error_msg}", "danger")
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
            except Exception as history_err:
                app.logger.warning(f"Failed to load model history: {history_err}")

        # Prepare data for template (store only essential data, not image)
        is_vlm = result.get("is_vlm", False)
        template_data = {
            "temp_image_file": temp_image_filename,
            "predicted_class": result.get("predicted_class", "Unknown"),
            "confidence_percent": int(result.get("confidence", 0) * 100),
            "probabilities": [
                (p["class_name"], p["probability"])
                for p in result.get("all_predictions", [])[:10]
            ],
            "model_arch": model_info.get("model_path", "Unknown"),
            "model_params": f"{model_info.get('total_params', 0):,}"
            if "total_params" in model_info
            else "Unknown",
            "rounds_completed": rounds_completed,
            "current_accuracy": f"{current_accuracy:.2f}",
            "is_vlm": is_vlm,
        }

        if is_vlm:
            template_data["vlm_description"] = result.get("vlm_description", {})
            template_data["raw_output"] = result.get("raw_output", "")

        app.logger.info(
            f"Prediction successful: {result['predicted_class']} ({result['confidence']:.2%})"
        )

        # Store results in session and redirect
        session["prediction_results"] = template_data
        return redirect(url_for("show_results"))

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}", exc_info=True)
        flash(f"Error processing image: {str(e)}", "danger")

        # Clean up file if it exists
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

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


@app.route("/api/models")
@login_required
def api_available_models():
    """Get list of available models"""
    models = get_available_models()
    classifier = get_classifier()
    current_model = classifier.current_model_type if classifier else "imagenet"

    return jsonify({"success": True, "models": models, "current_model": current_model})


@app.route("/api/switch_model", methods=["POST"])
@login_required
def api_switch_model():
    """Switch to a different model"""
    data = request.get_json()

    if not data or "model_type" not in data:
        return jsonify(
            {"success": False, "error": "Missing 'model_type' in request body"}
        ), 400

    model_type = data["model_type"]
    app.logger.info(f"Switching model to: {model_type}")

    result = switch_model(model_type)

    if result["success"]:
        app.logger.info(f"Successfully switched to {model_type} model")
        return jsonify(result)
    else:
        app.logger.error(f"Failed to switch model: {result.get('error')}")
        return jsonify(result), 500


@app.route("/api/current_model")
@login_required
def api_current_model():
    """Get current model info"""
    classifier = get_classifier()
    model_info = classifier.get_model_info()

    return jsonify(
        {
            "success": True,
            "current_model": classifier.current_model_type,
            "model_info": model_info,
        }
    )


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
