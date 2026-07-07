"""
SignVoice - hand landmark extraction.

Runs MediaPipe Hands over a dataset of sign-language class folders (0-9, A-Z)
and produces:
  * X.npy         - float32, shape (N, 126) landmark vectors
  * y.npy         - int64,   shape (N,)     class labels
  * label_map.json - {"0": 0, ..., "A": 10, ..., "Z": 35}

126 = 21 landmarks x 2 hands x 3 coords (x, y, z). Missing hand is zero-padded.
"""

import json
import os
import string

import cv2
import mediapipe as mp
import numpy as np


# Where the raw dataset lives. Each subfolder = one class label.
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPTS_DIR
LABEL_MAP_PATH = os.path.join(SCRIPTS_DIR, "label_map.json")

NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
NUM_HANDS = 2
PER_HAND_SIZE = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK          # 63
LANDMARK_VECTOR_SIZE = PER_HAND_SIZE * NUM_HANDS                       # 126

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [str(d) for d in range(10)] + list(string.ascii_uppercase)
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


mp_hands = mp.solutions.hands
_hands_detector = None


def _get_detector():
    """Lazy singleton — reused across images to avoid re-init overhead."""
    global _hands_detector
    if _hands_detector is None:
        _hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=NUM_HANDS,
            min_detection_confidence=0.5,
        )
    return _hands_detector


def _hand_to_vector(hand_landmarks):
    return np.array(
        [c for lm in hand_landmarks.landmark for c in (lm.x, lm.y, lm.z)],
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
    result = _get_detector().process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand_vectors = [
        _hand_to_vector(h) for h in result.multi_hand_landmarks[:NUM_HANDS]
    ]
    while len(hand_vectors) < NUM_HANDS:
        hand_vectors.append(np.zeros(PER_HAND_SIZE, dtype=np.float32))

    return np.concatenate(hand_vectors)


def _is_image_file(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def process_dataset(dataset_path, output_path):
    """Walk the dataset, extract landmarks, save X.npy + y.npy + label_map.json."""
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    os.makedirs(output_path, exist_ok=True)

    X = []
    y = []

    class_folders = sorted(
        name for name in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, name))
    )
    if not class_folders:
        print(f"No class folders found under {dataset_path}")
        return

    print(f"Found {len(class_folders)} class folders under {dataset_path}\n")

    for class_name in class_folders:
        if class_name not in LABEL_MAP:
            print(f"[skip] {class_name}: not in label map (expected 0-9, A-Z)")
            continue

        label = LABEL_MAP[class_name]
        class_dir = os.path.join(dataset_path, class_name)
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

    x_path = os.path.join(output_path, "X.npy")
    y_path = os.path.join(output_path, "y.npy")
    np.save(x_path, X_arr)
    np.save(y_path, y_arr)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)

    print(f"\nSaved: {x_path}  shape={X_arr.shape}")
    print(f"Saved: {y_path}  shape={y_arr.shape}")
    print(f"Saved: {LABEL_MAP_PATH}  ({len(LABEL_MAP)} classes)")


if __name__ == "__main__":
    process_dataset(DATASET_PATH, OUTPUT_DIR)
