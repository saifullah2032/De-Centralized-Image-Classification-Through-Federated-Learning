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
from frontend_web.inference import get_classifier, get_available_models


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
    classifier = get_classifier(mode="standard")
    model_info = classifier.get_model_info()
    return render_template(
        "predict.html",
        model_info=model_info,
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

    # Add recent analyses to results for comparison dashboard
    recent_analyses = session.get("recent_analyses", [])
    results["recent_analyses"] = recent_analyses

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

        # Get prediction mode from request
        mode = request.form.get("mode", "standard")  # "standard" or "ensemble"

        # Make prediction
        app.logger.info(f"Loading classifier for {mode} mode...")
        classifier = get_classifier(mode=mode)

        if mode == "ensemble":
            app.logger.info(f"Running hybrid inference on {filename}...")
            result = classifier.predict_hybrid(filepath)
        else:
            app.logger.info(f"Running standard inference on {filename}...")
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

        # Log detailed result structure for debugging
        app.logger.info(f"Prediction result keys: {list(result.keys())}")
        if mode == "ensemble":
            app.logger.info(f"Stage 1 data: {result.get('stage_1', {})}")
            app.logger.info(
                f"VLM Description keys: {list(result.get('vlm_description', {}).keys())}"
            )
            if result.get("vlm_description"):
                sample_key = (
                    list(result.get("vlm_description", {}).keys())[0]
                    if result.get("vlm_description")
                    else None
                )
                if sample_key:
                    app.logger.info(
                        f"Sample analysis ({sample_key}): {result.get('vlm_description', {}).get(sample_key, 'N/A')[:100]}"
                    )

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
        template_data = {
            "temp_image_file": temp_image_filename,
            "predicted_class": result.get("predicted_class", "Unknown"),
            "confidence_percent": int(result.get("confidence", 0) * 100),
            "model_arch": model_info.get("model_path", "Unknown"),
            "model_params": f"{model_info.get('total_params', 0):,}"
            if "total_params" in model_info
            else "Unknown",
            "rounds_completed": rounds_completed,
            "current_accuracy": f"{current_accuracy:.2f}",
            "mode": mode,
        }

        if mode == "ensemble":
            # Hybrid mode results (now Triple-Layer with SCL)
            # CRITICAL: Extract vlm_corrected_class from result if VLM self-corrected
            vlm_corrected_class = result.get("vlm_corrected_class")
            final_predicted_class = (
                vlm_corrected_class
                if vlm_corrected_class
                else result.get("predicted_class", "Unknown")
            )

            template_data.update(
                {
                    "is_hybrid": True,
                    "vlm_corrected_class": vlm_corrected_class,  # Pass to template for results display
                    "stage_1_results": result.get("stage_1", {}),
                    "stage_1_5_scl_results": result.get("stage_1_5_nuclear", {}),
                    "stage_2_results": {
                        "model": "BLIP-VQA",
                        "analysis_type": "Context-Aware 9-Point Visual Intelligence",
                        "analysis": result.get("vlm_description", {}),
                    },
                    "model_credits": "Identification powered by ImageNet-1K MobileNetV2 | Nuclear Truth Protocol SCL | Analysis powered by BLIP-VQA",
                }
            )
        else:
            # Standard mode results
            template_data.update(
                {
                    "is_hybrid": False,
                    "probabilities": [
                        (p["class_name"], p["probability"])
                        for p in result.get("all_predictions", [])[:10]
                    ],
                }
            )

        app.logger.info(
            f"Prediction successful ({mode} mode): {result['predicted_class']} ({result.get('confidence', 0):.2%})"
        )

        # Store results in session and redirect
        session["prediction_results"] = template_data

        # Add to recent analyses for comparison dashboard (keep last 3)
        recent_analyses = session.get("recent_analyses", [])

        # Extract confidence properly based on mode
        if mode == "ensemble":
            stage_1_data = result.get("stage_1", {})
            conf_percent = int(stage_1_data.get("confidence", 0) * 100)
            stage_1_pred = stage_1_data.get("predicted_class", "Unknown")
        else:
            conf_percent = int(result.get("confidence", 0) * 100)
            stage_1_pred = ""

        recent_analysis_entry = {
            "temp_image_file": temp_image_filename,
            "predicted_class": result.get("predicted_class", "Unknown"),
            "confidence_percent": conf_percent,
            "mode": mode,
            "stage_1_prediction": stage_1_pred,
        }
        recent_analyses.insert(0, recent_analysis_entry)  # Add to front
        session["recent_analyses"] = recent_analyses[:3]  # Keep only last 3

        app.logger.info(f"Added to recent_analyses: {recent_analysis_entry}")

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


@app.route("/predict_hybrid", methods=["POST"])
@login_required
def predict_hybrid_image():
    """
    Handle hybrid CNN+VLM prediction via dedicated endpoint
    This is kept for backward compatibility but redirects to standard predict with mode=ensemble
    """
    # Redirect to /predict with ensemble mode
    app.logger.info(
        "Hybrid prediction request - redirecting to /predict with ensemble mode..."
    )

    # Get file and process similarly but set mode to ensemble
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

        # Load classifier in ensemble mode
        app.logger.info(f"Loading classifier for hybrid (ensemble) prediction...")
        classifier = get_classifier(mode="ensemble")

        app.logger.info(f"Running hybrid inference on {filename}...")
        result = classifier.predict_hybrid(filepath)

        # Save image temporarily
        temp_image_filename = filename

        # Check if prediction was successful
        if not result.get("success"):
            error_msg = result.get(
                "error", "Unknown error occurred during hybrid prediction"
            )
            app.logger.error(f"Hybrid prediction failed: {error_msg}")
            flash(f"Hybrid prediction failed: {error_msg}", "danger")
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

        # Prepare data for template
        stage1_data = result.get("stage_1", {})
        stage2_data = result.get("vlm_description", {})

        # CRITICAL: Extract vlm_corrected_class if VLM self-corrected
        vlm_corrected_class = result.get("vlm_corrected_class")
        final_predicted_class = (
            vlm_corrected_class
            if vlm_corrected_class
            else result.get("predicted_class", "Unknown")
        )

        template_data = {
            "temp_image_file": temp_image_filename,
            "predicted_class": final_predicted_class,  # Use corrected class if VLM triggered self-correction
            "vlm_corrected_class": vlm_corrected_class,  # Pass to template for results display
            "confidence_percent": int(stage1_data.get("confidence", 0) * 100),
            "confidence": stage1_data.get("confidence", 0),
            "model_arch": model_info.get("model_path", "Unknown"),
            "model_params": f"{model_info.get('total_params', 0):,}"
            if "total_params" in model_info
            else "Unknown",
            "rounds_completed": rounds_completed,
            "current_accuracy": f"{current_accuracy:.2f}",
            "is_hybrid": True,
            "mode": "ensemble",
            "stage_1_results": {
                "model": stage1_data.get("model", "MobileNetV2"),
                "dataset": stage1_data.get("dataset", "ImageNet-1K"),
                "predicted_class": stage1_data.get("predicted_class", "Unknown"),
                "confidence": stage1_data.get("confidence", 0),
                "confidence_percent": stage1_data.get("confidence_percent", "0%"),
                "routing_mode": stage1_data.get("routing_mode", "Standard"),
                "context_used": stage1_data.get("context_used", False),
            },
            "stage_2_results": {
                "model": "BLIP-VQA",
                "analysis_type": "Context-Aware 9-Point Visual Intelligence",
                "analysis": stage2_data,
            },
            "model_credits": "Identification powered by ImageNet-1K MobileNetV2 | Analysis powered by BLIP-VQA",
        }

        app.logger.info(
            f"Hybrid prediction successful: {result.get('predicted_class')} "
            f"(CNN confidence: {stage1_data.get('confidence', 0):.2%})"
        )

        # Store results in session and redirect
        session["prediction_results"] = template_data

        # Add to recent analyses for comparison dashboard (keep last 3)
        recent_analyses = session.get("recent_analyses", [])

        # Extract confidence as integer percentage
        conf_percent_str = stage1_data.get("confidence_percent", "0%")
        if isinstance(conf_percent_str, str):
            conf_percent = int(conf_percent_str.rstrip("%"))
        else:
            conf_percent = int(stage1_data.get("confidence", 0) * 100)

        recent_analysis_entry = {
            "temp_image_file": temp_image_filename,
            "predicted_class": stage1_data.get("predicted_class", "Unknown"),
            "confidence_percent": conf_percent,
            "mode": "ensemble",
            "stage_1_prediction": stage1_data.get("predicted_class", ""),
        }
        recent_analyses.insert(0, recent_analysis_entry)  # Add to front
        session["recent_analyses"] = recent_analyses[:3]  # Keep only last 3

        app.logger.info(f"Added to recent_analyses (hybrid): {recent_analysis_entry}")

        return redirect(url_for("show_results"))

    except Exception as e:
        app.logger.error(
            f"Error processing image for hybrid prediction: {str(e)}", exc_info=True
        )
        flash(f"Error processing image for hybrid prediction: {str(e)}", "danger")

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
    """Get list of available modes (only standard and ensemble)"""
    classifier = get_classifier(mode="standard")

    return jsonify(
        {
            "success": True,
            "modes": {
                "standard": {
                    "name": "Standard Mode",
                    "description": "ImageNet-1K Classification",
                    "model": "MobileNetV2",
                    "dataset": "ImageNet-1K (1,000 classes)",
                },
                "ensemble": {
                    "name": "Ensemble Mode",
                    "description": "ImageNet-1K + BLIP-VQA Hybrid Analysis",
                    "stage_1": "MobileNetV2 (ImageNet-1K)",
                    "stage_2": "BLIP-VQA (Visual Question Answering)",
                },
            },
            "current_mode": classifier.mode,
        }
    )


@app.route("/api/switch_model", methods=["POST"])
@login_required
def api_switch_model():
    """
    Switch between standard and ensemble modes
    Note: Only CIFAR-100 model is used, we're switching between prediction modes
    """
    data = request.get_json()

    if not data or "mode" not in data:
        return jsonify(
            {
                "success": False,
                "error": "Missing 'mode' in request body. Use 'standard' or 'ensemble'",
            }
        ), 400

    mode = data.get("mode", "standard")

    if mode not in ["standard", "ensemble"]:
        return jsonify(
            {
                "success": False,
                "error": f"Invalid mode: {mode}. Use 'standard' or 'ensemble'",
            }
        ), 400

    app.logger.info(f"Switching to {mode} mode...")

    try:
        classifier = get_classifier(mode=mode)
        model_info = classifier.get_model_info()

        app.logger.info(f"Successfully switched to {mode} mode")
        return jsonify({"success": True, "mode": mode, "model_info": model_info})
    except Exception as e:
        app.logger.error(f"Failed to switch mode: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/current_model")
@login_required
def api_current_model():
    """Get current model/mode info"""
    classifier = get_classifier(mode="standard")
    model_info = classifier.get_model_info()

    return jsonify(
        {
            "success": True,
            "current_mode": classifier.mode,
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


@app.route("/privacy-protocol")
def privacy_protocol():
    """Display privacy protocol and safeguards documentation"""
    return render_template("privacy_protocol.html")


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.route("/download_csv")
@login_required
def download_csv():
    """
    Download prediction results as CSV file
    Exports the complete Triple-Layer Consistency Pipeline audit including:
    - Stage 1 CNN classification metadata
    - Stage 1.5 SCL verification status and interrogative check
    - Stage 3 VLM 9-point professional analysis
    """
    import csv
    from io import StringIO

    # Get results from session
    results = session.get("prediction_results")

    if not results:
        flash("No prediction results found. Please run a prediction first.", "warning")
        return redirect(url_for("predict"))

    try:
        # Create CSV in memory
        csv_buffer = StringIO()
        csv_writer = csv.writer(csv_buffer)

        # Write header
        csv_writer.writerow(
            ["Technical Audit Report - Triple-Layer Consistency Pipeline"]
        )
        csv_writer.writerow(
            ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        )
        csv_writer.writerow([])

        # STAGE 1: CNN CLASSIFICATION METADATA
        csv_writer.writerow(["STAGE 1: CNN CLASSIFICATION (MobileNetV2)"])
        stage_1 = results.get("stage_1_results", {})
        csv_writer.writerow(
            ["Predicted Class", stage_1.get("predicted_class", "Unknown")]
        )
        csv_writer.writerow(["Confidence (%)", stage_1.get("confidence_percent", "0%")])
        csv_writer.writerow(["Confidence (Raw)", f"{stage_1.get('confidence', 0):.4f}"])
        csv_writer.writerow(["Routing Mode", stage_1.get("routing_mode", "Standard")])
        csv_writer.writerow(
            ["Context Used", "Yes" if stage_1.get("context_used") else "No"]
        )
        csv_writer.writerow(["Model Type", stage_1.get("model", "MobileNetV2")])
        csv_writer.writerow(["Dataset", stage_1.get("dataset", "ImageNet-1K")])
        csv_writer.writerow([])

        # STAGE 1.5: SEMANTIC CONSISTENCY LAYER (SCL) - NEW
        scl_results = results.get("stage_1_5_scl_results", {})
        if scl_results:
            csv_writer.writerow(["STAGE 1.5: SEMANTIC CONSISTENCY LAYER (SCL)"])
            csv_writer.writerow(["SCL Status", scl_results.get("scl_status", "N/A")])
            csv_writer.writerow(
                ["VLM Verified", "Yes" if scl_results.get("scl_verified") else "No"]
            )
            csv_writer.writerow(
                [
                    "Interrogative Question",
                    scl_results.get("interrogative_question", "N/A"),
                ]
            )
            csv_writer.writerow(
                ["VLM Response", scl_results.get("scl_response", "N/A")]
            )
            csv_writer.writerow([])

        # STAGE 3: VLM 9-POINT PROFESSIONAL ANALYSIS
        csv_writer.writerow(["STAGE 3: VLM ANALYSIS (BLIP-VQA) - 9-POINT SYNTHESIS"])
        csv_writer.writerow(["Index", "Category", "Synthesized Analysis"])

        # Define all 9 categories
        categories = [
            "Common Identity",
            "Visual Summary",
            "Operational Utility",
            "Provenance & Setting",
            "Technical Nomenclature",
            "Safety & Risk Assessment",
            "Maintenance & Longevity",
            "Aesthetic & Design Style",
            "Interaction & Relationship",
        ]

        # Try to get analysis from stage_2_results first, then from vlm_description
        analysis = {}
        if results.get("is_hybrid"):
            stage_2_results = results.get("stage_2_results", {})
            analysis = stage_2_results.get("analysis", {})

        # If analysis is still empty, try vlm_description
        if not analysis:
            vlm_desc = results.get("vlm_description", {})
            if vlm_desc:
                analysis = vlm_desc

        # Always write all 9 categories, whether hybrid or standard mode
        for idx, category in enumerate(categories, 1):
            insight = analysis.get(category, "No data available")
            csv_writer.writerow([idx, category, insight])

        csv_writer.writerow([])

        # DEEP-DATA EXPORT: 9-POINT TECHNICAL SUMMARY (Column Format)
        csv_writer.writerow(["DEEP-DATA EXPORT: TECHNICAL AUDIT COLUMNS"])
        csv_writer.writerow(
            [
                "Export Format",
                "Timestamp-based technical columns for integration and analysis",
            ]
        )
        csv_writer.writerow([])

        # Define categories for deep-data export
        categories = [
            "Common Identity",
            "Visual Summary",
            "Operational Utility",
            "Provenance & Setting",
            "Technical Nomenclature",
            "Safety & Risk Assessment",
            "Maintenance & Longevity",
            "Aesthetic & Design Style",
            "Interaction & Relationship",
        ]

        # Create column headers for deep-data export
        deep_data_headers = [
            "Timestamp",
            "Stage 1 Result",
            "Confidence %",
            "SCL Verification Status",
        ] + categories
        csv_writer.writerow(deep_data_headers)

        # Build data row with all available information
        stage_1 = results.get("stage_1_results", {})
        scl_results = results.get("stage_1_5_scl_results", {})

        # Get analysis from stage_2_results or vlm_description
        analysis = {}
        if results.get("is_hybrid"):
            stage_2_results = results.get("stage_2_results", {})
            analysis = stage_2_results.get("analysis", {})

        if not analysis:
            vlm_desc = results.get("vlm_description", {})
            if vlm_desc:
                analysis = vlm_desc

        # Build data row with all fields
        deep_data_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
            stage_1.get("predicted_class", "Unknown"),  # Stage 1 Result
            stage_1.get("confidence_percent", "0%"),  # Confidence %
            scl_results.get("scl_status", "N/A"),  # SCL Verification Status
        ]

        # Add all 9-point analyses
        for category in categories:
            insight = analysis.get(category, "No data available")
            deep_data_row.append(insight)

        csv_writer.writerow(deep_data_row)

        csv_writer.writerow([])

        # SYSTEM INTELLIGENCE METADATA
        csv_writer.writerow(["SYSTEM INTELLIGENCE"])
        csv_writer.writerow(["Pipeline Type", "Triple-Layer Consistency Pipeline"])
        csv_writer.writerow(["Stage 1 Model", "ImageNet-1K MobileNetV2"])
        csv_writer.writerow(["Stage 1.5 Verification", "BLIP-VQA Interrogative Check"])
        csv_writer.writerow(["Stage 3 Model", "BLIP-VQA 9-Point Synthesis"])
        csv_writer.writerow(["Inference Speed (Stage 1)", "~50-200ms"])
        csv_writer.writerow(["Inference Speed (Stage 1.5)", "~500-800ms"])
        csv_writer.writerow(["Inference Speed (Stage 3)", "~1-3 seconds per image"])
        csv_writer.writerow(["Hardware Profile", "CPU Optimized (8GB RAM mode)"])
        csv_writer.writerow(["Processing Mode", "8GB RAM Optimized (CPU)"])
        csv_writer.writerow([])

        # PREDICTION MODE & CREDITS
        csv_writer.writerow(["METADATA"])
        csv_writer.writerow(["Prediction Mode", results.get("mode", "Unknown")])
        csv_writer.writerow(
            ["Model Architecture", results.get("model_arch", "Unknown")]
        )
        csv_writer.writerow(
            ["Federated Learning Rounds", results.get("rounds_completed", 0)]
        )
        csv_writer.writerow([])

        # FOOTER
        csv_writer.writerow(
            [results.get("model_credits", "Decentralized Multimodal Visual Assistant")]
        )
        csv_writer.writerow(["Report Type", "Complete Technical Audit"])
        csv_writer.writerow(
            ["Pipeline Version", "Triple-Layer Consistency Pipeline v3.0"]
        )
        csv_writer.writerow(
            ["Data Completeness", "Full 9-point synthesis with SCL verification"]
        )

        # Create response
        csv_data = csv_buffer.getvalue()

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=audit_report_triple_layer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            },
        )

    except Exception as e:
        app.logger.error(f"Error generating CSV: {str(e)}")
        flash(f"Error generating CSV: {str(e)}", "danger")
        return redirect(url_for("show_results"))


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
