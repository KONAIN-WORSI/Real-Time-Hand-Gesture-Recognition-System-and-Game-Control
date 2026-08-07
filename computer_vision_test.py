import cv2
import mediapipe as mp

# 1. Access the MediaPipe Hands solution
mp_hands = mp.solutions.hands if hasattr(mp, 'solutions') else mp.python.solutions.hands
mp_draw = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else mp.python.solutions.drawing_utils

# 2. Initialize the Hand Detector object
hands = mp_hands.Hands(
    static_image_mode=False,     # False = treating as continuous video, not isolated images
    max_num_hands=2,            # Track 1 hand for maximum CPU speed
    min_detection_confidence=0.6 # Minimum confidence threshold to register a hand
)

# 3. Open webcam (0 is usually default laptop camera)
cap = cv2.VideoCapture(0)

print("Starting Webcam Test... Press 'q' in the window to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not accessible.")
        break

    # Flip the frame horizontally for a natural 'mirror' effect
    frame = cv2.flip(frame, 1)

    # OpenCV reads frames in BGR format; MediaPipe requires RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame through MediaPipe model
    results = hands.process(rgb_frame)

    # Check if a hand was detected in the current frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw visual lines and keypoints on top of original BGR frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Render frame in window
    cv2.imshow("Module 1 Test: Vision Pipeline", frame)

    # Stop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()