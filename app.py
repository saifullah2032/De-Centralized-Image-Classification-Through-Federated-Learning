import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.preprocessing import image

# --- App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fed_secret_key_2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), default='client') # 'admin' or 'client'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ML Model Architecture ---
def create_model():
    """Returns a MobileNetV2 model for CIFAR-10 classification."""
    model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_pw = generate_password_hash(request.form['password'], method='pbkdf2:sha256')
        new_user = User(username=request.form['username'], password=hashed_pw, role=request.form['role'])
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return render_template('server_dashboard.html', user=current_user)
    return render_template('client_dashboard.html', user=current_user)

@app.route('/train_client', methods=['POST'])
@login_required
def train_client():
    """
    This route triggers the local training. In a real scenario, this would 
    start the flwr.client.start_numpy_client() process.
    """
    dataset_type = request.form.get('dataset')
    # Trigger logic for client training here
    flash(f'Started Federated Training using {dataset_type} partition.')
    return redirect(url_for('dashboard'))

@app.route('/classify', methods=['GET', 'POST'])
@login_required
def classify():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 1. Load Model with Global Weights
        model = create_model()
        weights_path = "global_model_weights.h5"
        
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            
            # 2. Image Preprocessing
            img = image.load_img(filepath, target_size=(32, 32))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # 3. Prediction
            preds = model.predict(img_array)
            result_idx = np.argmax(preds[0])
            confidence = np.max(preds[0]) * 100
            
            return render_template('results.html', 
                                   label=CLASSES[result_idx], 
                                   confidence=round(confidence, 2),
                                   img_path=filepath)
        else:
            flash("Global model weights not found. Train the model first!")
            return redirect(url_for('dashboard'))

    return render_template('classify.html')

import time

@app.route('/stream-status')
def stream_status():
    def generate():
        while True:
            if os.path.exists("train_status.txt"):
                with open("train_status.txt", "r") as f:
                    status = f.read()
                yield f"data: {status}\n\n"
            else:
                yield "data: Waiting for training to start...\n\n"
            time.sleep(2) # Check every 2 seconds
            
    return app.response_class(generate(), mimetype='text/event-stream')

# --- Initialization ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Creates database.db and tables
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
    app.run(debug=True, port=5000)