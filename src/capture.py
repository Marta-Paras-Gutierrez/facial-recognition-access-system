"""
=========================================================
Facial Recognition System
Module: capture.py

Description:
This module registers a new user by capturing facial images from the webcam.
The images are automatically cropped and stored in the dataset folder.

Workflow:
1. Ask for the person's name.
2. Create a personal folder if it does not exist.
3. Open the webcam.
4. Detect the face.
5. Save cropped face images.
6. Stop when ESC is pressed or the maximum number
   of images has been captured.
=========================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import cv2					# OpenCV library for computer vision

# Import configuration constants
from config import (
    CAMERA_INDEX,
    SCALE_FACTOR,
    MIN_NEIGHBORS,
    MAX_IMAGES,
    SAVE_EVERY,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    WINDOW_CAPTURE
)

# Import utility functions
from utils import (
    print_title,
    load_face_detector,
    ensure_directory,
    DATASET_PATH
)

# =============================================================================
# FUNCTION: capture_faces()
# =============================================================================


def capture_faces():

    print("\n========================================")
    print(" Register New User")
    print("========================================")

    # -------------------------------------------------------------------------
    # Ask for the person's name
    # -------------------------------------------------------------------------

    person_name = input("Enter person's name: ").strip()

    if not person_name:
        print("\n[ERROR] Name cannot be empty.\n")
        return

    # -------------------------------------------------------------------------
    # Create folder for this user
    # -------------------------------------------------------------------------

    person_folder = DATASET_PATH / person_name
    person_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load face detector
    # -------------------------------------------------------------------------

    face_detector = load_face_detector()

    # -------------------------------------------------------------------------
    # Open webcam
    # -------------------------------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("\n[ERROR] Unable to open the webcam.\n")
        return

    print("\nCamera started successfully.")
    print("Move your head slowly in different directions.")
    print("Press ESC to stop early.\n")

    # -------------------------------------------------------------------------
    # Capture variables
    # -------------------------------------------------------------------------

    image_counter = 0
    frame_counter = 0

    # -------------------------------------------------------------------------
    # Capture loop
    # -------------------------------------------------------------------------

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        # Mirror effect for a more natural interaction
        frame = cv2.flip(frame, 1)

        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS
        )

        frame_counter += 1

        for (x, y, w, h) in faces:

            # Draw rectangle around the detected face
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Save face every N frames
            if frame_counter % SAVE_EVERY == 0:

                face = gray[y:y + h, x:x + w]

                face = cv2.resize(
                    face,
                    (IMAGE_WIDTH, IMAGE_HEIGHT),
                    interpolation=cv2.INTER_CUBIC
                )

                filename = person_folder / f"face_{image_counter:03}.jpg"

                cv2.imwrite(str(filename), face)

                image_counter += 1

            # Display captured images counter
            cv2.putText(
                frame,
                f"Captured: {image_counter}/{MAX_IMAGES}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(WINDOW_CAPTURE, frame)

        key = cv2.waitKey(1)

        if key == 27 or image_counter >= MAX_IMAGES:
            break

    # -------------------------------------------------------------------------
    # Release resources
    # -------------------------------------------------------------------------

    camera.release()
    cv2.destroyAllWindows()

    print("\n========================================")
    print(f"User: {person_name}")
    print(f"Images captured: {image_counter}")
    print("Registration completed successfully.")
    print("========================================\n")


# =============================================================================
# TEST MODULE
# =============================================================================

if __name__ == "__main__":
    capture_faces()
