"""
=========================================================
Facial Recognition System
Module: train.py

Description:
This module trains a facial recognition model using the
images stored in the dataset folder.

Workflow:
1. Scan all registered users.
2. Read every face image.
3. Assign a numeric label to each user.
4. Train the LBPH recognizer.
5. Save the trained model.
=========================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import cv2                          # OpenCV library for computer vision
import numpy as np                  # Numerical operations

# Import configuration constants
from config import LBPH_THRESHOLD

# Import utility functions
from utils import (
    print_title,
    get_registered_users,
    count_images,
    DATASET_PATH,
    MODELS_PATH,
    MODEL_FILE,
    ensure_directory
)

# =============================================================================
# FUNCTION: train_model()
# =============================================================================


def train_model():

    print("\n========================================")
    print(" Training LBPH Model")
    print("========================================")

    # -------------------------------------------------------------------------
    # Get registered users
    # -------------------------------------------------------------------------

    users = get_registered_users()

    if len(users) == 0:

        print("\n[ERROR] No registered users found in dataset.")
        print("Please register at least one user first.\n")
        return

    # -------------------------------------------------------------------------
    # Dataset statistics
    # -------------------------------------------------------------------------

    total_images = count_images(DATASET_PATH)

    print("\nScanning dataset...\n")

    labels = []
    faces = []

    # -------------------------------------------------------------------------
    # Read every user's images
    # -------------------------------------------------------------------------

    for label, user_folder in enumerate(users):

        images = list(user_folder.glob("*.jpg"))

        print(f"{user_folder.name:<15} {len(images)} images")

        for image_path in images:

            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(label)

    print("\n----------------------------------------")
    print(f"Total users : {len(users)}")
    print(f"Total images: {total_images}")
    print("----------------------------------------\n")

    if len(faces) == 0:

        print("[ERROR] No valid images found.\n")
        return

    # -------------------------------------------------------------------------
    # Create LBPH recognizer
    # -------------------------------------------------------------------------

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Training model...\n")

    recognizer.train(
        faces,
        np.array(labels)
    )

    # -------------------------------------------------------------------------
    # Save trained model
    # -------------------------------------------------------------------------

    recognizer.write(str(MODEL_FILE))

    print("========================================")
    print(" Training completed successfully")
    print("========================================")

    print(f"\nModel saved at:\n{MODEL_FILE}\n")

    # -------------------------------------------------------------------------
    # Save labels mapping
    # -------------------------------------------------------------------------

    labels_file = MODELS_PATH / "labels.txt"

    with open(labels_file, "w", encoding="utf-8") as file:

        for label, user in enumerate(users):

            file.write(f"{label}:{user.name}\n")

    print("Labels file generated successfully.\n")


# =============================================================================
# TEST MODULE
# =============================================================================

if __name__ == "__main__":
    train_model()
