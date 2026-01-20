from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import tensorflow as tf

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), default='client') # 'admin' or 'client'

def create_model():
    model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), weights=None, classes=10)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model