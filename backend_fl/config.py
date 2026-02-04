"""
Configuration file for Federated Learning system
Contains constants and hyperparameters for model training and system setup
"""

import os
from dotenv import load_dotenv

# Set Keras backend to JAX (before any keras imports)
os.environ["KERAS_BACKEND"] = "jax"

# Load environment variables
load_dotenv()

# Dataset Configuration
DATASET = os.getenv("DATASET", "CIFAR100")  # CIFAR10 or CIFAR100
INPUT_SHAPE = (32, 32, 3)  # CIFAR image dimensions
NUM_CLASSES = 100 if DATASET == "CIFAR100" else 10  # 100 for CIFAR-100, 10 for CIFAR-10

# CIFAR-10 Labels (10 classes)
CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# CIFAR-100 Fine Labels (100 classes)
CIFAR100_FINE_LABELS = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

# CIFAR-100 Coarse Labels (20 superclasses)
CIFAR100_COARSE_LABELS = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
]

# Mapping of fine labels to coarse labels (superclass grouping)
CIFAR100_FINE_TO_COARSE = {
    # aquatic_mammals (0)
    4: 0,
    30: 0,
    55: 0,
    72: 0,
    95: 0,  # beaver, dolphin, otter, seal, whale
    # fish (1)
    1: 1,
    32: 1,
    67: 1,
    73: 1,
    91: 1,  # aquarium_fish, flatfish, ray, shark, trout
    # flowers (2)
    54: 2,
    62: 2,
    70: 2,
    82: 2,
    92: 2,  # orchid, poppy, rose, sunflower, tulip
    # food_containers (3)
    9: 3,
    10: 3,
    16: 3,
    28: 3,
    61: 3,  # bottle, bowl, can, cup, plate
    # fruit_and_vegetables (4)
    0: 4,
    51: 4,
    53: 4,
    57: 4,
    83: 4,  # apple, mushroom, orange, pear, sweet_pepper
    # household_electrical_devices (5)
    22: 5,
    39: 5,
    40: 5,
    86: 5,
    87: 5,  # clock, keyboard, lamp, telephone, television
    # household_furniture (6)
    5: 6,
    20: 6,
    25: 6,
    84: 6,
    94: 6,  # bed, chair, couch, table, wardrobe
    # insects (7)
    6: 7,
    7: 7,
    14: 7,
    18: 7,
    24: 7,  # bee, beetle, butterfly, caterpillar, cockroach
    # large_carnivores (8)
    3: 8,
    42: 8,
    43: 8,
    88: 8,
    97: 8,  # bear, leopard, lion, tiger, wolf
    # large_man-made_outdoor_things (9)
    12: 9,
    17: 9,
    37: 9,
    68: 9,
    76: 9,  # bridge, bus, house, road, skyscraper
    # large_natural_outdoor_scenes (10)
    23: 10,
    33: 10,
    49: 10,
    60: 10,
    71: 10,  # cloud, forest, mountain, plain, sea
    # large_omnivores_and_herbivores (11)
    15: 11,
    19: 11,
    21: 11,
    31: 11,
    38: 11,  # camel, cattle, chimpanzee, elephant, kangaroo
    # medium_mammals (12)
    34: 12,
    63: 12,
    64: 12,
    66: 12,
    75: 12,  # fox, possum, raccoon, skunk, squirrel
    # non-insect_invertebrates (13)
    26: 13,
    45: 13,
    77: 13,
    79: 13,
    99: 13,  # crab, lobster, snail, spider, worm
    # people (14)
    2: 14,
    11: 14,
    35: 14,
    46: 14,
    98: 14,  # baby, boy, girl, man, woman
    # reptiles (15)
    27: 15,
    29: 15,
    44: 15,
    78: 15,
    93: 15,  # crocodile, dinosaur, lizard, snake, turtle
    # small_mammals (16)
    36: 16,
    50: 16,
    65: 16,
    74: 16,
    80: 16,  # hamster, mouse, rabbit, shrew, squirrel
    # trees (17)
    47: 17,
    52: 17,
    56: 17,
    59: 17,
    96: 17,  # maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
    # vehicles_1 (18)
    8: 18,
    13: 18,
    48: 18,
    58: 18,
    90: 18,  # bicycle, bus, motorcycle, pickup_truck, train
    # vehicles_2 (19)
    41: 19,
    69: 19,
    81: 19,
    85: 19,
    89: 19,  # lawn_mower, rocket, streetcar, tank, tractor
}

# Use appropriate labels based on dataset
LABELS = CIFAR100_FINE_LABELS if DATASET == "CIFAR100" else CIFAR10_LABELS

# Training Hyperparameters
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", 10))  # Number of federated learning rounds
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", 5))  # Number of federated clients
LOCAL_EPOCHS = int(
    os.getenv("LOCAL_EPOCHS", 5)
)  # Local training epochs per round (increased from 3)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))  # Training batch size (increased from 32)
LEARNING_RATE = float(
    os.getenv("LEARNING_RATE", 0.0005)
)  # Model learning rate (reduced for stability)

# Model Architecture Configuration
MODEL_ARCHITECTURE = os.getenv(
    "MODEL_ARCHITECTURE", "enhanced_mobilenet"
)  # Model type to use
MODEL_ALPHA = float(
    os.getenv("MODEL_ALPHA", 1.0)
)  # MobileNet width multiplier (increased from 0.5)
USE_PRETRAINED = (
    os.getenv("USE_PRETRAINED", "false").lower() == "true"
)  # Use ImageNet weights

# Data Augmentation Configuration
USE_AUGMENTATION = (
    os.getenv("USE_AUGMENTATION", "true").lower() == "true"
)  # Enable data augmentation
AUGMENTATION_STRATEGY = os.getenv(
    "AUGMENTATION_STRATEGY", "standard"
)  # Augmentation type

# Non-IID Data Configuration
ALPHA = 0.5  # Dirichlet distribution parameter for Non-IID partitioning
# Lower alpha = more heterogeneous data distribution

# Federated Learning Server Configuration
FL_SERVER_HOST = os.getenv(
    "FL_SERVER_HOST", "0.0.0.0"
)  # Server binds to all interfaces
FL_SERVER_PORT = int(os.getenv("FL_SERVER_PORT", 8080))
FL_SERVER_ADDRESS = f"{FL_SERVER_HOST}:{FL_SERVER_PORT}"  # Server bind address
FL_CLIENT_SERVER_ADDRESS = os.getenv(
    "FL_CLIENT_SERVER_ADDRESS", "localhost:8080"
)  # Client connection address

# Flask Web Server Configuration
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", 5000))
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# Model Paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/global_model.h5")
MODEL_HISTORY_PATH = "models/model_history.json"

# Upload Configuration
UPLOAD_FOLDER = os.path.abspath("uploads")  # Use absolute path for send_from_directory
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE", 5 * 1024 * 1024))  # 5 MB

# Logging Configuration
LOG_DIR = "logs"
TRAINING_LOG_PATH = os.path.join(LOG_DIR, "training.log")
PRIVACY_REPORT_PATH = os.path.join(LOG_DIR, "privacy_report.json")

# Security Configuration
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", 3600))  # 1 hour

# Strategy Configuration
MIN_AVAILABLE_CLIENTS = 2  # Minimum clients needed to start training
MIN_FIT_CLIENTS = 2  # Minimum clients needed for each round
FRACTION_FIT = 1.0  # Fraction of clients to sample for training
FRACTION_EVALUATE = 1.0  # Fraction of clients to sample for evaluation

# Performance Thresholds
TARGET_ACCURACY = 0.85  # Target global model accuracy
INFERENCE_TIMEOUT = 0.5  # Maximum inference time in seconds
AGGREGATION_TIMEOUT = 15  # Maximum aggregation time in seconds

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("tests/data", exist_ok=True)

print(f"Configuration loaded successfully!")
print(f"  - Dataset: {DATASET} ({NUM_CLASSES} classes)")
print(f"  - FL Server: {FL_SERVER_ADDRESS}")
print(f"  - Web Server: {WEB_HOST}:{WEB_PORT}")
print(
    f"  - Training: {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients, {LOCAL_EPOCHS} local epochs"
)
print(f"  - Non-IID Alpha: {ALPHA}")
