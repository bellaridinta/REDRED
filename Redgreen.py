"""
REDRED — Gesture Recognition Camera
===================================
- Launches the webcam.
- Cross your hands in front of the camera  -> screen gets a RED   shade (50%).
- Raise both open palms up in the air       -> screen gets a GREEN shade (50%).

Controls:
    ESC or 'q'  -> quit

Requires: opencv-contrib-python, mediapipe (>=0.10, Tasks API), numpy
Also requires the model file `hand_landmarker.task` next to this script.
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# Landmark indices for fingertips and the joint just below them (PIP).
FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]
WRIST = 0

# Bones to draw so the hand looks like a skeleton overlay.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def count_extended_fingers(landmarks):
    """Return how many of the 4 main fingers are pointing up (extended)."""
    extended = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        # In image coordinates, smaller y means higher up on the screen.
        if landmarks[tip].y < landmarks[pip].y:
            extended += 1
    return extended


def is_open_palm(landmarks):
    """An open palm = at least 4 fingers extended."""
    return count_extended_fingers(landmarks) >= 4


def is_metal(landmarks):
    """
    Metal / rock 'horns' sign: index and pinky up, middle and ring folded.
    """
    index_up = landmarks[8].y < landmarks[6].y
    pinky_up = landmarks[20].y < landmarks[18].y
    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    return index_up and pinky_up and middle_down and ring_down


def palm_cross_z(landmarks):
    """
    Signed area (z of the 2D cross product) of the palm triangle
    wrist -> index_mcp(5) and wrist -> pinky_mcp(17).
    The sign flips when the hand turns front<->back. It is invariant to
    in-plane rotation, so finger direction doesn't matter.
    """
    wrist, idx, pky = landmarks[0], landmarks[5], landmarks[17]
    ax, ay = idx.x - wrist.x, idx.y - wrist.y
    bx, by = pky.x - wrist.x, pky.y - wrist.y
    return ax * by - ay * bx


def is_front_palm(landmarks, label):
    """
    True only when an open palm is facing the camera (not the back of hand).
    Uses the palm winding (palm_cross_z) together with handedness.
    """
    if not is_open_palm(landmarks):
        return False
    cz = palm_cross_z(landmarks)
    if label == "Right":
        return cz < 0
    if label == "Left":
        return cz > 0
    return False


def apply_shade(frame, color, alpha=0.5):
    """Blend a solid color over the frame at `alpha` transparency."""
    overlay = np.full_like(frame, color, dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def draw_centered_label(frame, text, scale=3.0, thickness=6):
    """Draw big text centered on the frame (white with a black outline)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    x = (w - tw) // 2
    y = (h + th) // 2
    # Black outline for readability over any shade, then white fill.
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0),
                thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def detect_gesture(hands):
    """
    Decide the gesture from the detected hands.

    `hands` is a list of (landmarks, label) tuples, label being "Left"/"Right".
    Returns one of: "red", "green", or None.
    """
    if len(hands) < 2:
        return None

    (h1, l1), (h2, l2) = hands[0], hands[1]

    # RED: both hands open palms FACING the camera (not the back of the hand).
    if is_front_palm(h1, l1) and is_front_palm(h2, l2):
        return "red"

    # GREEN: both hands making the metal / rock 'horns' sign.
    if is_metal(h1) and is_metal(h2):
        return "green"

    return None


def draw_hand(frame, landmarks):
    """Draw landmark points and bones for one hand."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    NEON_GREEN = (20, 255, 57)   # BGR
    YELLOW = (0, 255, 255)       # BGR
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], NEON_GREEN, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, YELLOW, -1)


def build_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model file not found: {MODEL_PATH}")
        return

    detector = build_detector()

    cap = cv2.VideoCapture(0)          # 0 = default webcam
    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
        return

    # BGR colors (OpenCV uses Blue-Green-Red order).
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Failed to read a frame from the camera.")
                break

            # Mirror the frame so it feels like a selfie view.
            frame = cv2.flip(frame, 1)

            # MediaPipe Tasks expects an mp.Image in RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start) * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            hands_landmarks = result.hand_landmarks or []
            handedness = result.handedness or []
            hands = []
            for i, landmarks in enumerate(hands_landmarks):
                draw_hand(frame, landmarks)
                label = handedness[i][0].category_name if i < len(handedness) else "?"
                hands.append((landmarks, label))

            gesture = detect_gesture(hands)

            if gesture == "red":
                frame = apply_shade(frame, RED)
                draw_centered_label(frame, "RED RED")
            elif gesture == "green":
                frame = apply_shade(frame, GREEN)
                draw_centered_label(frame, "GREEN GREEN")
            else:
                cv2.putText(
                    frame, "CORTIS - REDRED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame, "ESC / q to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                cv2.LINE_AA,
            )

            cv2.imshow("REDRED - Gesture Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):   # ESC or q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
