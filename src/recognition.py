"""
=========================================================
Facial Recognition System
Module: recognition.py

Description:
This module performs real-time facial recognition using
the previously trained LBPH model.

Workflow:
1. Load the trained model.
2. Load the labels file.
3. Open the webcam.
4. Detect faces.
5. Predict the user's identity.
6. Display the recognition result.
=========================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import cv2                          # OpenCV library for computer vision

# Import configuration constants
from config import (
    CAMERA_INDEX,
    SCALE_FACTOR,
    MIN_NEIGHBORS,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    LBPH_THRESHOLD,
    WINDOW_RECOGNITION
)

# Import utility functions
from utils import (
    print_title,
    load_face_detector,
    load_model,
    load_labels
)

# =============================================================================
# FUNCTION: start_recognition()
# =============================================================================


def start_recognition():

    print("\n========================================")
    print(" Real-Time Facial Recognition")
    print("========================================")

	# -------------------------------------------------------------------------
    # Load required components
    # -------------------------------------------------------------------------

    try:
        face_detector = load_face_detector()
        recognizer = load_model()
        labels = load_labels()

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        return

    # -------------------------------------------------------------------------
    # Open webcam
    # -------------------------------------------------------------------------

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():

        print("\n[ERROR] Unable to open webcam.\n")
        return

    print("\nRecognition started.")
    print("Press ESC to exit.\n")

    # -------------------------------------------------------------------------
    # Recognition loop
    # -------------------------------------------------------------------------

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS
        )

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                interpolation=cv2.INTER_CUBIC
            )

            label, confidence = recognizer.predict(face)

            if confidence < LBPH_THRESHOLD:
                person = labels.get(label, "Unknown")
                color = (0, 255, 0)
                status = "Recognized"

            else:
                person = "Unknown"
                color = (0, 0, 255)
                status = "Unknown"

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # User name
            cv2.putText(
                frame,
                person,
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # Confidence
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

			# Console output
            print(
                f"Status: {status:11} | "
                f"User: {person:15} | "
                f"Confidence: {confidence:.2f}",
                end="\r"
            )

        cv2.imshow(WINDOW_RECOGNITION, frame)

        key = cv2.waitKey(1)

        if key == 27:
            break

    # -------------------------------------------------------------------------
    # Release resources
    # -------------------------------------------------------------------------

    camera.release()
    cv2.destroyAllWindows()

    print("\n")
    print("========================================")
    print(" Recognition finished")
    print("========================================\n")


# =============================================================================
# TEST MODULE
# =============================================================================

if __name__ == "__main__":
    start_recognition()
