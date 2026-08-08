import cv2
import mediapipe as mp
import pandas as pd

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Memory list to store extracted dataset rows
dataset = []

print("--- DATASET COLLECTOR READY ---")
print("Hold a gesture in front of the camera and PRESS & HOLD keys:")
print("  '0' = Fist (Mute)")
print("  '1' = Open Palm (Play/Pause)")
print("  '2' = Thumbs Up (Volume Up)")
print("  '3' = Peace Sign (Volume Down)")
print("  '4' = Hand Zero (Brightness Down)")
print("  '5' = Hand One  (Brightness Up)")
print("Press 'q' when finished to save gestures.csv.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Read active keypress from user
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    active_label = None
    if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
        active_label = int(chr(key))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw green skeletal overlay
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If a valid gesture key ('0'-'5') is pressed, extract features
            if active_label is not None:
                # Get Wrist Origin (Joint 0)
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                row_features = []
                # Extract 21 relative coordinate pairs
                for lm in hand_landmarks.landmark:
                    norm_x = lm.x - wrist_x
                    norm_y = lm.y - wrist_y
                    row_features.extend([norm_x, norm_y])

                # Full Row = [label, coord_0, coord_1, ..., coord_41]
                full_row = [active_label] + row_features
                dataset.append(full_row)
                print(f"Logged Row #{len(dataset)} for Gesture Class '{active_label}'")

    # Show live webcam
    cv2.imshow("Module 3: Data Collection", frame)

cap.release()
cv2.destroyAllWindows()

# Save collected rows to CSV
if dataset:
    # Create column headers
    columns = ["label"] + [f"coord_{i}" for i in range(42)]
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv("gestures.csv", index=False)
    print(f"\n SUCCESS: Saved {len(dataset)} rows to 'gestures.csv'!")
else:
    print("\n No data was recorded.")