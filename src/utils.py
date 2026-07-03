"""
=========================================================
Facial Recognition System
Module: utils.py

Description:
This module contains helper functions shared across the
project.

Keeping common functionality in one place avoids code
duplication and makes maintenance easier.
=========================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import cv2
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset folder
DATASET_PATH = PROJECT_ROOT / "data" / "dataset"

MODELS_PATH = PROJECT_ROOT / "data" / "models"

# Haar Cascade classifier
CASCADE_PATH = PROJECT_ROOT / "src" / "Encuadre_de_las_caras.xml"

MODEL_FILE = MODELS_PATH / "lbph_model.xml"

LABELS_FILE = MODELS_PATH / "labels.txt"

# =============================================================================
# PRINT TITLE
# =============================================================================


def print_title(title: str):

    print("\n========================================")
    print(f" {title}")
    print("========================================")


# =============================================================================
# LOAD FACE DETECTOR
# =============================================================================


def load_face_detector():

    detector = cv2.CascadeClassifier(str(CASCADE_PATH))

    if detector.empty():
        raise FileNotFoundError(
            f"Haar Cascade not found:\n{CASCADE_PATH}"
        )

    return detector


# =============================================================================
# LOAD LABELS
# =============================================================================


def load_labels():

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            "labels.txt not found."
        )

    labels = {}

    with open(LABELS_FILE, "r", encoding="utf-8") as file:

        for line in file:

            label, name = line.strip().split(":")

            labels[int(label)] = name

    return labels


# =============================================================================
# LOAD LBPH MODEL
# =============================================================================


def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "LBPH model not found."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(str(MODEL_FILE))

    return recognizer


# =============================================================================
# ENSURE DIRECTORY EXISTS
# =============================================================================


def ensure_directory(path: Path):

    path.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# GET REGISTERED USERS
# =============================================================================


def get_registered_users():

    if not DATASET_PATH.exists():
        return []

    return sorted(
        [
            folder
            for folder in DATASET_PATH.iterdir()
            if folder.is_dir()
        ]
    )


# =============================================================================
# COUNT DATASET IMAGES
# =============================================================================


def count_images(dataset_path: Path):

    total = 0

    for person in dataset_path.iterdir():

        if person.is_dir():

            total += len(
                list(
                    person.glob("*.jpg")
                )
            )

    return total