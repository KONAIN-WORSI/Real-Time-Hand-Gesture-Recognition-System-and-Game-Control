import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
data = []

print("Press '0' for Fist, '1' for Open Palm, '2' for Thumbs Up, '3' for Peace. Press 'q' to Quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    label_to_save = None
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
        label_to_save = int(chr(key))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if label_to_save is not None:
                # Extract coordinates
                landmarks = []
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                # Feature Engineering: Normalize all coordinates relative to wrist (index 0)
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - wrist_x, lm.y - wrist_y])

                row = [label_to_save] + landmarks
                data.append(row)
                print(f"Logged sample for gesture: {label_to_save} (Total: {len(data)})")

    cv2.imshow("Dataset Collector", frame)

cap.release()
cv2.destroyAllWindows()

# Save collected landmarks to CSV
if data:
    columns = ["label"] + [f"coord_{i}" for i in range(42)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("gestures.csv", index=False)
    print("Dataset saved to gestures.csv successfully!")