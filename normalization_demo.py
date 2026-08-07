import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands if hasattr(mp, 'solutions') else mp.python.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

print("Starting Normalization Demo... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Raw Wrist Coordinates (Joint 0)
            raw_wrist_x = hand_landmarks.landmark[0].x
            raw_wrist_y = hand_landmarks.landmark[0].y

            # 2. Raw Index Fingertip Coordinates (Joint 8)
            raw_index_x = hand_landmarks.landmark[8].x
            raw_index_y = hand_landmarks.landmark[8].y

            # 3. Apply Wrist-Centric Normalization
            norm_index_x = raw_index_x - raw_wrist_x
            norm_index_y = raw_index_y - raw_wrist_y

            # Print comparison to terminal
            print(f"RAW Index (Joint 8): ({raw_index_x:.2f}, {raw_index_y:.2f})  |  NORMALIZED Index relative to Wrist: ({norm_index_x:.2f}, {norm_index_y:.2f})")

    cv2.imshow("Module 2: Normalization Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()