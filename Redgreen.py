import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# Landmark for fingertips and the joint 
FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]
WRIST = 0

# Bones to draw so the hand looks like a skeleton overlay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

#Return how many of the 4 main fingers are pointing up
def count_extended_fingers(landmarks):
    extended = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip].y < landmarks[pip].y:
            extended += 1
    return extended

#Open palm hand sign
def is_open_palm(landmarks):
    return count_extended_fingers(landmarks) >= 4 #at least four fingers up

#Metal hand sign
def is_metal(landmarks):
    index_up = landmarks[8].y < landmarks[6].y
    pinky_up = landmarks[20].y < landmarks[18].y
    middle_down = landmarks[12].y > landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    return index_up and pinky_up and middle_down and ring_down

#Palm crossing sign
def palm_cross_z(landmarks):
    wrist, idx, pky = landmarks[0], landmarks[5], landmarks[17]
    ax, ay = idx.x - wrist.x, idx.y - wrist.y
    bx, by = pky.x - wrist.x, pky.y - wrist.y
    return ax * by - ay * bx

#Define the front side of the palm only
def is_front_palm(landmarks, label):
    if not is_open_palm(landmarks):
        return False
    cz = palm_cross_z(landmarks)
    if label == "Right":
        return cz < 0
    if label == "Left":
        return cz > 0
    return False

#Add shade with 50% transparency
def apply_shade(frame, color, alpha=0.5):
    overlay = np.full_like(frame, color, dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

#Add text at the center of the frame
def draw_centered_label(frame, text, scale=3.0, thickness=6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    x = (w - tw) // 2
    y = (h + th) // 2
    # Add black outline
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0),
                thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255),
                thickness, cv2.LINE_AA)

#Decide the gesture from the detected hands
def detect_gesture(hands):
    if len(hands) < 2:
        return None

    (h1, l1), (h2, l2) = hands[0], hands[1]

    # RED: both hands open palms facing the camera (not the back of the hand)
    if is_front_palm(h1, l1) and is_front_palm(h2, l2):
        return "red"

    # GREEN: both hands making the metal sign
    if is_metal(h1) and is_metal(h2):
        return "green"

    return None

#Draw landmark points and bones for hands
def draw_hand(frame, landmarks):
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


    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Failed to read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)

            
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
