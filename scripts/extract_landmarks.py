"""
SignVoice - hand landmark extraction from static images.

Runs MediaPipe Hands over a dataset laid out as:
  data/train/<class>/*.jpg
  data/val/<class>/*.jpg
  data/test/<class>/*.jpg

and produces one (X, y) pair per split under scripts/:
  * X_train.npy / y_train.npy
  * X_val.npy   / y_val.npy
  * X_test.npy  / y_test.npy
  * label_map.json - {"0": 0, ..., "A": 10, ..., "Z": 35}

Each image yields a single 126-value vector
(21 landmarks x 2 hands x 3 coords). Missing hand is zero-padded.
"""

import json
import os
import string

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# EDIT THIS: path to the dataset root (must contain train/ val/ test/ subfolders).
# Example for Google Colab: DATASET_PATH = "/content/data"
# ---------------------------------------------------------------------------
DATASET_PATH = "data/isl_dataset_split"

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "data/landmarks"
LABEL_MAP_PATH = os.path.join(SCRIPTS_DIR, "label_map.json")
MODEL_PATH = os.path.join(SCRIPTS_DIR, "hand_landmarker.task")

NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
NUM_HANDS = 2
PER_HAND_SIZE = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK          # 63
LANDMARK_VECTOR_SIZE = PER_HAND_SIZE * NUM_HANDS                       # 126

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [str(d) for d in range(10)] + list(string.ascii_uppercase)
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

SPLITS = ("train", "val", "test")


_hands_detector = None


def _get_detector():
    """Lazy singleton — reused across images to avoid re-init overhead."""
    global _hands_detector
    if _hands_detector is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {MODEL_PATH}. "
                "Download from https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.5,
        )
        _hands_detector = mp_vision.HandLandmarker.create_from_options(options)
    return _hands_detector


def _hand_to_vector(hand_landmarks):
    return np.array(
        [c for lm in hand_landmarks for c in (lm.x, lm.y, lm.z)],
        dtype=np.float32,
    )


def extract_landmarks_from_image(image_path):
    """Return a flat 126-value vector, or None if no hands are detected.

    Raises IOError if the image cannot be read/decoded.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_detector().detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand_vectors = [
        _hand_to_vector(h) for h in result.hand_landmarks[:NUM_HANDS]
    ]
    while len(hand_vectors) < NUM_HANDS:
        hand_vectors.append(np.zeros(PER_HAND_SIZE, dtype=np.float32))

    return np.concatenate(hand_vectors)


def _is_image_file(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def _process_split(split_dir, split_name):
    """Extract (X, y) arrays for a single split folder."""
    X = []
    y = []

    if not os.path.isdir(split_dir):
        print(f"[skip] split '{split_name}': folder not found at {split_dir}")
        return (
            np.empty((0, LANDMARK_VECTOR_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    class_folders = sorted(
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )
    if not class_folders:
        print(f"No class folders found under {split_dir}")
        return (
            np.empty((0, LANDMARK_VECTOR_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    print(f"\n== split '{split_name}' — {len(class_folders)} class folders ==")

    for class_name in class_folders:
        if class_name not in LABEL_MAP:
            print(f"[skip] {class_name}: not in label map (expected 0-9, A-Z)")
            continue

        label = LABEL_MAP[class_name]
        class_dir = os.path.join(split_dir, class_name)
        image_files = [
            f for f in sorted(os.listdir(class_dir)) if _is_image_file(f)
        ]

        extracted = 0
        skipped = 0
        for image_name in image_files:
            image_path = os.path.join(class_dir, image_name)
            try:
                vector = extract_landmarks_from_image(image_path)
            except IOError as exc:
                print(f"  [unreadable] {image_path}: {exc}")
                skipped += 1
                continue

            if vector is None:
                skipped += 1
                continue

            X.append(vector)
            y.append(label)
            extracted += 1

        total = extracted + skipped
        print(
            f"[{class_name:>2}] label={label:>2}  "
            f"extracted={extracted:>4}  skipped={skipped:>4}  total={total:>4}"
        )

    X_arr = (
        np.array(X, dtype=np.float32) if X
        else np.empty((0, LANDMARK_VECTOR_SIZE), dtype=np.float32)
    )
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr


def process_dataset(dataset_path, output_path):
    """Process train/val/test splits and save one (X, y) pair per split."""
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    os.makedirs(output_path, exist_ok=True)

    for split_name in SPLITS:
        split_dir = os.path.join(dataset_path, split_name)
        X_arr, y_arr = _process_split(split_dir, split_name)

        x_path = os.path.join(output_path, f"X_{split_name}.npy")
        y_path = os.path.join(output_path, f"y_{split_name}.npy")
        np.save(x_path, X_arr)
        np.save(y_path, y_arr)

        print(f"\nSaved: {x_path}  shape={X_arr.shape}")
        print(f"Saved: {y_path}  shape={y_arr.shape}")

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"\nSaved: {LABEL_MAP_PATH}  ({len(LABEL_MAP)} classes)")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, OUTPUT_DIR)
